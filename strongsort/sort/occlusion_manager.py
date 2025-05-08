import numpy as np


class OcclusionManager:
    """Manager for handling occlusions between tracks"""

    def __init__(self):
        self.occluded_tracks = {}  # Dict to store occluded tracks: {track_id: (occluding_track_id, occlusion_score)}
        self.occlusion_history = {}  # Dict to track occlusion history
        self.track_trajectories = {}  # Store recent trajectories for prediction
        self.max_trajectory_length = 20

    def detect_occlusions(self, tracks):
        """Detect occlusions between active tracks"""
        # Reset current occlusion states
        current_occlusions = {}

        # Check all pairs of active tracks
        active_tracks = [t for t in tracks if t.is_confirmed()]

        for i, track1 in enumerate(active_tracks):
            for j, track2 in enumerate(active_tracks[i + 1:], i + 1):
                # Calculate IoU between bounding boxes
                bbox1 = track1.to_tlbr()
                bbox2 = track2.to_tlbr()
                iou_score = self._calculate_iou(bbox1, bbox2)

                # If significant overlap, consider it an occlusion
                if iou_score > 0.45:
                    # Determine which track is occluding which based on depth (size can be a proxy)
                    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

                    # Use track age to help determine persistence
                    # If one track is much older, it's likely the one to keep
                    age_factor = min(1.0, abs(track1.age - track2.age) / 10.0)
                    confidence_factor = min(1.0, abs(track1.conf - track2.conf))

                    # The larger track is typically closer to the camera (occluding)
                    # Also consider track age and confidence in deciding
                    if (area1 > area2 and track1.age >= track2.age * 0.8) or (track1.age > track2.age * 2):
                        occluder, occluded = track1, track2
                    elif (area2 > area1 and track2.age >= track1.age * 0.8) or (track2.age > track1.age * 2):
                        occluder, occluded = track2, track1
                    else:
                        # If sizes are similar, use confidence as tiebreaker
                        if track1.conf > track2.conf * 1.2:
                            occluder, occluded = track1, track2
                        elif track2.conf > track1.conf * 1.2:
                            occluder, occluded = track2, track1
                        else:
                            # If all metrics are similar, consider higher track ID as occluder
                            # (typically means it's a newer track)
                            if track1.track_id > track2.track_id:
                                occluder, occluded = track1, track2
                            else:
                                occluder, occluded = track2, track1

                    # Record the occlusion
                    current_occlusions[occluded.track_id] = (occluder.track_id, iou_score)

                    # Mark the track as occluded
                    occluded.is_occluded = True
                    occluded.occluded_by = occluder.track_id

        # Update occlusion history
        for track_id, (occluder_id, score) in current_occlusions.items():
            if track_id not in self.occlusion_history:
                self.occlusion_history[track_id] = []

            self.occlusion_history[track_id].append((occluder_id, score))

            # Limit history size
            if len(self.occlusion_history[track_id]) > 30:  # 30 frames history
                self.occlusion_history[track_id].pop(0)

        # Update current occlusions
        self.occluded_tracks = current_occlusions

        return current_occlusions

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes in format [x1, y1, x2, y2]"""
        # Determine intersection rectangle
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])

        # No intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate area of both bounding boxes
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Calculate IoU
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

        return iou

    def update_track_trajectories(self, tracks):
        """Update the trajectory history for each active track"""
        for track in tracks:
            if track.is_confirmed():
                if track.track_id not in self.track_trajectories:
                    self.track_trajectories[track.track_id] = []

                # Add current position to trajectory
                self.track_trajectories[track.track_id].append((track.mean[:4].copy(), track.motion_mode))

                # Limit trajectory length
                if len(self.track_trajectories[track.track_id]) > self.max_trajectory_length:
                    self.track_trajectories[track.track_id].pop(0)

    def predict_occluded_tracks(self, tracks):
        """Predict the state of occluded tracks that weren't detected"""
        # First update trajectories with current track states
        self.update_track_trajectories(tracks)

        # Identify tracks that are likely occluded but not detected
        occluded_track_ids = list(self.occluded_tracks.keys())

        # For each occluded track, adjust its parameters to handle occlusion better
        for track in tracks:
            if track.track_id in occluded_track_ids:
                # Mark track as occluded in its state
                track.is_occluded = True

                # Record occluder information
                occluder_id, _ = self.occluded_tracks[track.track_id]
                track.occluded_by = occluder_id

                # If we have trajectory history, we can use it to enhance prediction
                if track.track_id in self.track_trajectories and len(self.track_trajectories[track.track_id]) > 5:
                    trajectory = self.track_trajectories[track.track_id]

                    # Check if the track was moving consistently before occlusion
                    recent_motion_modes = [t[1] for t in trajectory[-5:]]

                    # If track was moving consistently in one direction,
                    # we can adjust the Kalman prediction weight
                    if recent_motion_modes.count('acceleration') >= 3:
                        # Track was accelerating, increase velocity component in prediction
                        track.mean[4:6] *= 1.1  # Boost x,y velocity
                    elif recent_motion_modes.count('deceleration') >= 3:
                        # Track was decelerating, decrease velocity component
                        track.mean[4:6] *= 0.9  # Reduce x,y velocity

    def resolve_occlusions(self, tracks, detections):
        """Resolve occlusions and adjust detection matching weights"""
        modified_detections = detections.copy()

        # For each detection, check if it might belong to a recently occluded track
        for i, detection in enumerate(modified_detections):
            bbox = detection.tlwh
            detection_box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

            for track in tracks:
                if track.is_occluded:
                    # Calculate how similar this detection is to the occluded track
                    # Appearance similarity
                    appearance_sim = track.get_appearance_similarity(detection.feature)

                    # Motion-based similarity (how well it matches predicted position)
                    track_box = track.to_tlbr()
                    spatial_iou = self._calculate_iou(detection_box, track_box)

                    # Combined similarity score
                    similarity = 0.7 * appearance_sim + 0.3 * spatial_iou

                    # If high similarity to an occluded track, boost the matching score
                    if similarity > 0.6:
                        # This detection is likely the reappearance of an occluded track
                        # We don't modify the detection directly but will use this in matching
                        if not hasattr(detection, 'occlusion_matches'):
                            detection.occlusion_matches = []

                        detection.occlusion_matches.append((track.track_id, similarity))

        return modified_detections

    def update_after_matching(self, tracks):
        """Update track states after matching phase"""
        # Clean up occlusion states for tracks that have been matched
        for track in tracks:
            if track.is_occluded and track.time_since_update == 0:
                # Track was occluded but got matched in this frame
                track.is_occluded = False
                track.occluded_by = None
                track.occlusion_count = 0
                track._max_age = track.original_max_age

                # Remove from occluded tracks list
                if track.track_id in self.occluded_tracks:
                    del self.occluded_tracks[track.track_id]
