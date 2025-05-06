# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    增强版: 自适应过程噪声和运动模式识别，专为马拉松场景设计
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        # 自适应参数
        self.acceleration_factor = 0.5  # 加速度影响因子
        self.velocity_history = []  # 速度历史记录
        self.history_max_size = 5  # 历史记录最大长度
        self.previous_velocity = None  # 上一时刻速度
        self.velocity_change_threshold = 0.3  # 速度变化阈值

        # 运动模式分类参数
        self.current_motion_mode = "normal"  # 默认为正常运动模式
        self.motion_modes = {
            "normal": 1.0,  # 正常跑步 - 基础噪声水平
            "acceleration": 1.5,  # 加速 - 提高速度噪声
            "deceleration": 1.2,  # 减速 - 提高位置噪声
            "turn": 1.3,  # 转弯 - 提高角度噪声
            "stop": 0.8  # 停止 - 降低速度噪声
        }

    def classify_motion_pattern(self, current_velocity):
        """
        基于速度历史识别当前运动模式
        """
        if self.previous_velocity is None:
            return "normal"

        # 计算速度变化
        velocity_delta = current_velocity - self.previous_velocity
        position_velocity_change = np.linalg.norm(velocity_delta[:2])  # 仅考虑x,y速度变化

        # 更新速度历史
        self.velocity_history.append(current_velocity[:2])  # 仅存储x,y速度
        if len(self.velocity_history) > self.history_max_size:
            self.velocity_history.pop(0)

        # 计算当前速度大小
        current_speed = np.linalg.norm(current_velocity[:2])

        # 运动模式识别逻辑
        if current_speed < 0.1:
            return "stop"
        elif position_velocity_change > self.velocity_change_threshold:
            # 判断是加速还是减速
            if np.dot(self.previous_velocity[:2], velocity_delta[:2]) > 0:
                return "acceleration"
            else:
                return "deceleration"

        # 检测转弯 - 通过判断速度方向变化
        if len(self.velocity_history) >= 2:
            prev_direction = self.velocity_history[-2] / (np.linalg.norm(self.velocity_history[-2]) + 1e-6)
            curr_direction = current_velocity[:2] / (np.linalg.norm(current_velocity[:2]) + 1e-6)
            direction_change = np.arccos(np.clip(np.dot(prev_direction, curr_direction), -1.0, 1.0))

            if direction_change > 0.3:  # 约17度
                return "turn"

        return "normal"

    def initiate(self, measurement):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[0],  # the center point x
            2 * self._std_weight_position * measurement[1],  # the center point y
            1 * measurement[2],  # the ratio of width/height
            2 * self._std_weight_position * measurement[3],  # the height
            10 * self._std_weight_velocity * measurement[0],
            10 * self._std_weight_velocity * measurement[1],
            0.1 * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))

        # 初始化自适应参数
        self.previous_velocity = mean_vel
        self.velocity_history = []
        self.current_motion_mode = "normal"

        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        # 提取当前速度分量
        current_velocity = mean[4:8]

        # 识别运动模式
        if self.previous_velocity is not None:
            self.current_motion_mode = self.classify_motion_pattern(current_velocity)

        # 根据运动模式获取噪声调整因子
        mode_factor = self.motion_modes.get(self.current_motion_mode, 1.0)

        # 计算加速度及其大小
        acceleration = np.zeros_like(current_velocity)
        if self.previous_velocity is not None:
            acceleration = current_velocity - self.previous_velocity
        self.previous_velocity = current_velocity.copy()

        # 计算速度变化率 (主要考虑x,y方向)
        velocity_change_rate = np.linalg.norm(acceleration[:2])

        # 动态调整噪声系数
        adaptive_factor = 1.0 + self.acceleration_factor * velocity_change_rate
        noise_factor = adaptive_factor * mode_factor

        # 调整标准差 - 根据当前状态和变化率动态调整
        std_pos = [
            self._std_weight_position * mean[0],
            self._std_weight_position * mean[1],
            1 * mean[2],
            self._std_weight_position * mean[3],
        ]

        # 速度不确定性 - 根据运动模式和速度变化进行调整
        if self.current_motion_mode == "acceleration":
            # 加速模式下，增加速度状态的不确定性
            std_vel = [
                self._std_weight_velocity * mean[0] * noise_factor,
                self._std_weight_velocity * mean[1] * noise_factor,
                0.1 * mean[2],
                self._std_weight_velocity * mean[3]
            ]
        elif self.current_motion_mode == "deceleration":
            # 减速模式下，更多考虑位置不确定性
            std_pos = [x * noise_factor for x in std_pos]
            std_vel = [
                self._std_weight_velocity * mean[0],
                self._std_weight_velocity * mean[1],
                0.1 * mean[2],
                self._std_weight_velocity * mean[3]
            ]
        elif self.current_motion_mode == "turn":
            # 转弯模式下，调整宽高比的不确定性
            std_vel = [
                self._std_weight_velocity * mean[0] * noise_factor,
                self._std_weight_velocity * mean[1] * noise_factor,
                0.1 * mean[2] * noise_factor,  # 增加宽高比速度不确定性
                self._std_weight_velocity * mean[3]
            ]
        else:
            # 正常或停止模式
            std_vel = [
                self._std_weight_velocity * mean[0],
                self._std_weight_velocity * mean[1],
                0.1 * mean[2],
                self._std_weight_velocity * mean[3]
            ]

            if self.current_motion_mode == "stop":
                # 停止模式 - 减小速度不确定性
                std_vel = [x * 0.5 for x in std_vel]

        # 创建过程噪声矩阵
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # 应用运动模型
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, confidence=0.0):
        """Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        confidence: float
            检测框置信度
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        # 调整测量噪声 - 根据运动模式和检测置信度
        mode_factor = self.motion_modes.get(self.current_motion_mode, 1.0)

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]

        # 根据置信度和运动模式调整噪声
        confidence_factor = max(0.1, 1.0 - confidence)  # 确保至少有一些噪声
        std = [s * confidence_factor * mode_factor for s in std]

        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement, confidence=0.0):
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        confidence: float
            检测框置信度
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        # 根据运动模式调整创新权重
        if self.current_motion_mode == "acceleration":
            # 加速模式下，更信任测量
            innovation_weight = 1.2
        elif self.current_motion_mode == "deceleration":
            # 减速模式下，适度信任测量
            innovation_weight = 1.1
        elif self.current_motion_mode == "turn":
            # 转弯模式下，更信任测量
            innovation_weight = 1.3
        else:
            # 正常模式，标准权重
            innovation_weight = 1.0

        # 应用加权创新
        innovation = innovation * innovation_weight

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)

        # 动态调整门控阈值 - 根据运动模式
        if not only_position:
            dims = 4  # 完整状态空间维度
            base_threshold = chi2inv95[dims]

            # 根据运动模式调整门控阈值
            if self.current_motion_mode == "acceleration":
                # 加速时使用更宽松的门控
                return squared_maha / (base_threshold * 1.3)
            elif self.current_motion_mode == "turn":
                # 转弯时使用更宽松的门控
                return squared_maha / (base_threshold * 1.4)

        return squared_maha
