#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

constexpr float eps = 1e-4;

Eigen::Matrix3f NormalizeRotation(const Eigen::Matrix3f &R) {
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(
      R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  return svd.matrixU() * svd.matrixV().transpose();
}

struct DRot {
  explicit DRot(const Eigen::Vector3d &phi) {
    float d = phi.norm();
    float d2 = phi.squaredNorm();
    Eigen::Matrix3f W = Sophus::SO3f::hat(phi);
    if (d < eps) {
      dR = Eigen::Matrix3f::Identity() + W;
      Jr = Eigen::Matrix3f::Identity();
    } else {
      dR = Eigen::Matrix3f::Identity() + W * sin(d) / d +
           W * W * (1.0f - cos(d)) / d2;
      Jr = Eigen::Matrix3f::Identity() - W * (1.0f - cos(d)) / d2 +
           W * W * (d - sin(d)) / (d2 * d);
    }
  }
  Eigen::Matrix3f dR;
  Eigen::Matrix3f Jr;
};

class Imu {
 public:
  struct ImuMeasurement {
    explicit ImuMeasurement(const Eigen::Vector3f &a, const Eigen::Vector3f &g,
                            float t)
        : a(a), g(g), t(t) {}
    Eigen::Vector3f a;
    Eigen::Vector3f g;
    float t;
  };

  struct Bias {
    explicit Bias(const Eigen::Vector3f &a, const Eigen::Vector3f &g)
        : a(a), g(g) {}
    Eigen::Vector3f a;
    Eigen::Vector3f g;
  };

  void IntegrateNewMeasurement(const Eigen::Vector3f &acc_measured,
                               const Eigen::Vector3f &gyro_measured, float dt) {
    measurements_.push_back(ImuMeasurement(acc_measured, gyro_measured, dt));

    Eigen::Vector3f acc = acc_measured - b_.a;
    Eigen::Vector3f gyro = gyro_measured - b_.g;

    // XXX: Record average acceleration and angular velocity

    dP_ = dP_ + dV_ * dt + 0.5 * dR_ * acc * dt * dt;
    dV_ = dV_ + dR_ * acc * dt;

    Eigen::Matrix3f acchat = Sophus::SO3f::hat(acc);

    Eigen::Matrix<float, 9, 9> A = Eigen::Matrix<float, 9, 9>::Identity();
    A.block<3, 3>(3, 0) = -dR_ * dt * acchat;
    A.block<3, 3>(6, 0) = -0.5f * dR_ * dt * dt * acchat;
    A.block<3, 3>(6, 3) = Eigen::DiagonalMatrix<float, 3>(dt, dt, dt);

    Eigen::Matrix<float, 9, 6> B = Eigen::Matrix<float, 9, 6>::Zero();
    B.block<3, 3>(3, 3) = dR_ * dt;
    B.block<3, 3>(6, 3) = 0.5f * dR_ * dt * dt;

    JPa_ = JPa_ + JVa_ * dt - 0.5f * dR_ * dt * dt;
    JPg_ = JPg_ + JVg_ * dt - 0.5f * dR_ * dt * dt * acchat * JRg_;
    JVa_ = JVa_ - dR_ * dt;
    JVg_ = JVg_ - dR_ * dt * acchat * JRg_;

    auto drot = DRot(gyro * dt);
    dR_ = NormalizeRotation(dR_ * drot.dR);
    A.block<3, 3>(0, 0) = drot.dR.transpose();
    B.block<3, 3>(0, 0) = drot.Jr * dt;

    C_.block<9, 9>(0, 0) =
        A * C_.block<9, 9>(0, 0) * A.transpose() + B * Nga_ * B.transpose();
    C_.block<6, 6>(9, 9) += NgaWalk_;

    JRg_ = drot.dR.transpose() * JRg_ - drot.Jr * dt;

    // XXX: Record total time
  }

 private:
  std::vector<ImuMeasurement> measurements_;
  Bias b_;
  Eigen::Matrix3f dR_;
  Eigen::Vector3f dV_, dP_;
  Eigen::Matrix3f JRg_, JVg_, JVa_, JPg_, JPa_;
  Eigen::Matrix<float, 9, 9> C_;
  Eigen::Matrix<float, 6, 6> Nga_;
  Eigen::Matrix<float, 6, 6> NgaWalk_;
};
