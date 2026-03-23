//! Mapping task — occupancy grid updates from pseudo-lidar + pose.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use bus::Bus;
use mapping::Mapper;

pub fn spawn_mapping_task(
    bus: Arc<Bus>,
    mapper: Arc<RwLock<Mapper>>,
) -> tokio::task::JoinHandle<()> {
    let bus_map       = Arc::clone(&bus);
    let mapper_map    = Arc::clone(&mapper);
    let mut rx_lidar  = bus.vision_pseudo_lidar.subscribe();
    let mut rx_pose_m = bus.slam_pose2d.subscribe();
    tokio::spawn(async move {
        info!("Mapping task started");
        loop {
            match rx_lidar.recv().await {
                Ok(scan) => {
                    let pose = *rx_pose_m.borrow_and_update();
                    let (delta, frontiers, stats) = {
                        let mut m = mapper_map.write().await;
                        m.update(&scan, &pose)
                    };
                    let _ = bus_map.map_grid_delta.send(delta);
                    let _ = bus_map.map_frontiers.send(frontiers);
                    let _ = bus_map.map_explored_stats.send(stats);
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("Mapping task lagged {n} scans");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    })
}
