# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import

import numpy as np

from strongsort.sort import iou_matching, kalman_filter, linear_assignment
from strongsort.sort.track import Track
from strongsort.sort.occlusion_manager import OcclusionManager


class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    GATING_THRESHOLD = np.sqrt(kalman_filter.chi2inv95[4])

    def __init__(self, metric, max_iou_distance=0.9, max_age=30, n_init=3, _lambda=0, ema_alpha=0.9, mc_lambda=0.995):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

        # Initialize occlusion manager
        self.occlusion_manager = OcclusionManager()

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def camera_update(self, previous_img, current_img):
        for track in self.tracks:
            track.camera_update(previous_img, current_img)

    def update(self, detections, classes, confidences):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Detect occlusions among current tracks
        self.occlusion_manager.detect_occlusions(self.tracks)

        # Predict occluded tracks
        self.occlusion_manager.predict_occluded_tracks(self.tracks)

        # Modify detections based on occlusion information
        modified_detections = self.occlusion_manager.resolve_occlusions(self.tracks, detections)

        # Run matching cascade with occlusion awareness
        matches, unmatched_tracks, unmatched_detections = self._enhanced_match(modified_detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                detections[detection_idx],
                classes[detection_idx],
                confidences[detection_idx]
            )

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self._initiate_track(
                detections[detection_idx],
                classes[detection_idx].item(),
                confidences[detection_idx].item()
            )

        # Update occlusion states after matching
        self.occlusion_manager.update_after_matching(self.tracks)

        # Clean up deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _enhanced_match(self, detections):
        """Enhanced matching strategy with occlusion awareness"""

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])

            cost_matrix = self.metric.distance(features, targets)

            # Apply standard gating
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix, tracks, dets, track_indices, detection_indices
            )

            # Apply occlusion-aware adjustments to cost matrix
            for i, track_idx in enumerate(track_indices):
                track = tracks[track_idx]

                # If track was occluded, adjust matching cost based on occlusion data
                if hasattr(track, 'is_occluded') and track.is_occluded:
                    for j, det_idx in enumerate(detection_indices):
                        # Check if this detection has been marked as a potential match for an occluded track
                        if hasattr(dets[det_idx], 'occlusion_matches'):
                            for occluded_id, similarity in dets[det_idx].occlusion_matches:
                                if track.track_id == occluded_id:
                                    # Reduce cost based on similarity to promote matching
                                    cost_matrix[i, j] *= max(0.4, 1.0 - similarity)

                # Adjust cost based on motion mode
                if track.motion_mode == "acceleration" or track.motion_mode == "turn":
                    # In rapid motion, more weight on appearance
                    cost_matrix[i, :] *= 0.9

                # Adjust cost based on track age (more established tracks have lower cost)
                if track.age > 10:
                    age_factor = min(0.8, max(0.5, 10.0 / track.age))
                    cost_matrix[i, :] *= age_factor

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features with enhanced matching
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks
        )

        # Find tracks that are occluded but unmatched
        occluded_tracks = [
            i for i in unmatched_tracks_a
            if hasattr(self.tracks[i], 'is_occluded') and self.tracks[i].is_occluded
        ]

        # Separate regular unmatched tracks from occluded ones
        regular_unmatched = [i for i in unmatched_tracks_a if i not in occluded_tracks]

        # Associate remaining tracks together with unconfirmed tracks using IOU
        iou_track_candidates = unconfirmed_tracks + [
            k for k in regular_unmatched if self.tracks[k].time_since_update == 1
        ]

        # First do normal IOU matching for regular tracks
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        # Now try matching occluded tracks with relaxed parameters
        if occluded_tracks:
            # Use more permissive IOU threshold for occluded tracks
            matches_c, unmatched_tracks_c, unmatched_detections = linear_assignment.min_cost_matching(
                iou_matching.iou_cost,
                min(0.95, self.max_iou_distance * 1.3),  # 30% more permissive
                self.tracks,
                detections,
                occluded_tracks,
                unmatched_detections,
            )

            # Combine all matches
            matches = matches_a + matches_b + matches_c
            unmatched_tracks = list(set(unmatched_tracks_b + unmatched_tracks_c))
        else:
            # If no occluded tracks, just combine regular matches
            matches = matches_a + matches_b
            unmatched_tracks = unmatched_tracks_b

        return matches, unmatched_tracks, unmatched_detections

    def _full_cost_metric(self, tracks, dets, track_indices, detection_indices):
        """
        This implements the full lambda-based cost-metric. However, in doing so, it disregards
        the possibility to gate the position only which is provided by
        linear_assignment.gate_cost_matrix(). Instead, I gate by everything.
        Note that the Mahalanobis distance is itself an unnormalised metric. Given the cosine
        distance being normalised, we employ a quick and dirty normalisation based on the
        threshold: that is, we divide the positional-cost by the gating threshold, thus ensuring
        that the valid values range 0-1.
        Note also that the authors work with the squared distance. I also sqrt this, so that it
        is more intuitive in terms of values.
        """
        # Compute First the Position-based Cost Matrix
        pos_cost = np.empty([len(track_indices), len(detection_indices)])
        msrs = np.asarray([dets[i].to_xyah() for i in detection_indices])
        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            pos_cost[row, :] = (
                    np.sqrt(self.kf.gating_distance(track.mean, track.covariance, msrs, False))
                    / self.GATING_THRESHOLD
            )

            # Apply motion mode adaptive gating
            if hasattr(track, 'motion_mode'):
                if track.motion_mode == "acceleration":
                    # Allow bigger position changes during acceleration
                    pos_cost[row, :] *= 0.85
                elif track.motion_mode == "turn":
                    # Allow bigger position changes during turns
                    pos_cost[row, :] *= 0.8
                elif track.motion_mode == "stop":
                    # Be more strict on position for stopped objects
                    pos_cost[row, :] *= 1.1

        pos_gate = pos_cost > 1.0

        # Now Compute the Appearance-based Cost Matrix
        app_cost = self.metric.distance(
            np.array([dets[i].feature for i in detection_indices]),
            np.array([tracks[i].track_id for i in track_indices]),
        )

        # Apply occlusion-aware weighting to appearance costs
        for i, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            if hasattr(track, 'is_occluded') and track.is_occluded:
                # Reduce appearance weight for occluded tracks
                app_cost[i, :] *= 0.9

        app_gate = app_cost > self.metric.matching_threshold

        # Calculate adaptive lambda based on track state
        adaptive_lambda = np.ones((len(track_indices), 1)) * self._lambda

        for i, track_idx in enumerate(track_indices):
            track = tracks[track_idx]

            # Adjust lambda based on motion mode
            if hasattr(track, 'motion_mode'):
                if track.motion_mode == "acceleration" or track.motion_mode == "turn":
                    # For fast motion, trust position less
                    adaptive_lambda[i] = max(0.2, self._lambda * 0.8)
                elif track.motion_mode == "stop":
                    # For stopped objects, trust position more
                    adaptive_lambda[i] = min(0.9, self._lambda * 1.2)

            # Adjust lambda based on occlusion status
            if hasattr(track, 'is_occluded') and track.is_occluded:
                # For occluded tracks, trust appearance more
                adaptive_lambda[i] = max(0.2, adaptive_lambda[i] * 0.7)

            # Adjust lambda based on feature quality
            if hasattr(track, 'feature_quality_scores') and track.feature_quality_scores:
                avg_quality = sum(track.feature_quality_scores) / len(track.feature_quality_scores)
                # Higher quality -> more trust in appearance
                if avg_quality > 0.8:
                    adaptive_lambda[i] = max(0.2, adaptive_lambda[i] * 0.9)

        # Now combine and threshold
        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

        for i in range(len(track_indices)):
            # Apply adaptive lambda for each track
            cost_matrix[i, :] = adaptive_lambda[i] * pos_cost[i, :] + (1 - adaptive_lambda[i]) * app_cost[i, :]

        # Gate based on both position and appearance
        cost_matrix[np.logical_or(pos_gate, app_gate)] = linear_assignment.INFTY_COST

        return cost_matrix

    def _match(self, detections):
        # We'll use the enhanced matching method instead
        return self._enhanced_match(detections)

    def _initiate_track(self, detection, class_id, conf):
        self.tracks.append(
            Track(
                detection.to_xyah(),
                self._next_id,
                class_id,
                conf,
                self.n_init,
                self.max_age,
                self.ema_alpha,
                detection.feature,
            )
        )
        self._next_id += 1
