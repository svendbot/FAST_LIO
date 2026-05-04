/**
 * @file smfom_compensate.hpp
 * @brief Per-particle motion compensation for FAST_LIO + SMFoM.
 *
 * Mirrors IMU_Processing.hpp::UndistortPcl's backward-pass warp loop
 * but reads particle i's pose history (from `kf.pose_history(i)`) and
 * frame-end state (`kf.get_x(i)`) instead of the single filter mean.
 *
 * Lives on the FAST_LIO side of the integration because the warp
 * depends on PCL types and FAST_LIO-specific point fields
 * (`curvature`-as-timestamp encoding, lidar-IMU extrinsics inside the
 * state). SMFoM stays PCL-free.
 *
 * Call once per particle per scan, after UndistortPcl has built the
 * per-particle pose history. See INTEGRATION.md §5.2.
 */

#ifndef SMFOM_COMPENSATE_HPP
#define SMFOM_COMPENSATE_HPP

#include <algorithm>

#include <Eigen/Dense>

#include "common_lib.h"   // PointCloudXYZI, time_list, M3D, V3D
#include "so3_math.h"     // Exp(angvel, dt)
#include <use-smfom.hpp>  // esekfom::esekf, sesmfom::PoseSample

namespace sesmfom {

/**
 * @brief Warp `raw` against particle `particle_idx`'s pose history.
 *
 * Backward pass through `kf.pose_history(particle_idx)`, mirroring the
 * loop in IMU_Processing.hpp:311-345. The frame-end pose (target frame
 * for compensation) is `kf.get_x(particle_idx)` — this particle's
 * scan-end view.
 *
 * Pre: `raw` need not be sorted; this function copies and sorts by
 * point timestamp internally.
 *
 * Post: returned cloud's points are warped into the per-particle
 * scan-end IMU frame, ready to be consumed by `h_share_model` for
 * particle `particle_idx`.
 *
 * @tparam Filter        esekfom::esekf<...> (template-deduced).
 * @param  kf            Filter holding the per-particle pose history.
 * @param  particle_idx  Which particle to compensate against.
 * @param  raw           Pre-warp scan.
 * @return               A fresh PointCloudXYZI with warped points.
 */
template<typename Filter>
PointCloudXYZI compensate_for(
    Filter& kf,
    unsigned particle_idx,
    const PointCloudXYZI& raw)
{
    PointCloudXYZI out = raw;
    if (out.points.empty()) return out;

    std::sort(out.points.begin(), out.points.end(), time_list);

    const auto& x_end   = kf.get_x(particle_idx);
    const auto& history = kf.pose_history(particle_idx);

    // History needs at least 2 ticks for a backward pass over intervals.
    // Single-entry histories (no IMU ticks within the scan) leave the
    // cloud unwarped — same behaviour as IMUpose with one element.
    if (history.size() < 2) return out;

    auto it_pcl = out.points.end() - 1;
    for (auto it_kp = history.end() - 1; it_kp != history.begin(); --it_kp) {
        const PoseSample& head = *(it_kp - 1);
        const PoseSample& tail = *it_kp;

        const M3D&             R_imu   = head.rot;
        const Eigen::Vector3d& vel_imu = head.vel;
        const Eigen::Vector3d& pos_imu = head.pos;
        const Eigen::Vector3d& acc_imu = tail.acc;
        const Eigen::Vector3d& angvel  = tail.gyr;

        for (; it_pcl->curvature / 1000.0 > head.offset_time; --it_pcl) {
            const double dt = it_pcl->curvature / 1000.0 - head.offset_time;

            const M3D R_i(R_imu * Exp(angvel, dt));
            const V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
            const V3D T_ei(pos_imu + vel_imu * dt
                                   + 0.5 * acc_imu * dt * dt - x_end.pos);
            const V3D P_comp = x_end.offset_R_L_I.conjugate() * (
                x_end.rot.conjugate() * (
                    R_i * (x_end.offset_R_L_I * P_i + x_end.offset_T_L_I) + T_ei
                ) - x_end.offset_T_L_I);

            it_pcl->x = static_cast<float>(P_comp(0));
            it_pcl->y = static_cast<float>(P_comp(1));
            it_pcl->z = static_cast<float>(P_comp(2));

            if (it_pcl == out.points.begin()) break;
        }
    }

    return out;
}

}  // namespace sesmfom

#endif  // SMFOM_COMPENSATE_HPP
