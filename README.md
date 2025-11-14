# CDS Compute Cluster

## Overview

The **CDS Compute Cluster** is a high-performance computing environment developed by the Cornell Data Science team to provide local, on-demand compute resources for machine learning and data processing. It connects heterogeneous nodes, some with NVIDIA GPUs, into a unified Slurm-orchestrated cluster. This internal system eliminates cloud costs while enabling GPU-accelerated workloads, distributed training, and large-scale data processing.

## Architecture

* **Head Nodes and Compute Nodes:** The system includes a dedicated head node for job scheduling and multiple compute nodes with varied CPU and GPU configurations.
* **Networking:** All nodes are connected via static IPs on a private LAN. The head node bridges this internal network with campus Wi-Fi.
* **Shared Storage:** A shared NFS volume ensures consistent file access across all nodes.
* **Authentication and Sync:** Munge provides inter-node authentication, and Chrony ensures time synchronization, which is critical for job coordination.
* **Containerized Environments:** Docker is used to create uniform runtime environments across nodes, ensuring compatibility and reproducibility regardless of hardware or operating system differences.

## Key Technologies

* **Slurm Scheduler:** Manages job queuing and parallel dispatch across CPUs and GPUs.
* **GPU Scheduling:** Slurm tracks and allocates GPUs using GRES, supporting exclusive access and multi-GPU jobs.
* **Container Support:** Docker is used for consistent environments across different node architectures.

## Features

* **Batch and Interactive Jobs:** Users submit via `sbatch` or launch live sessions with `srun`.
* **Resource-Aware Scheduling:** Slurm dispatches based on node specifications, including GPU count and memory.
* **Scalable and Modular:** New nodes can be added with minimal configuration, and setup is fully documented.

## Engineering Highlights

* Built entirely in-house using open-source tools.
* Supports heterogeneous hardware with unified scheduling and storage.
* Enables real-world machine learning workflows like LLM inference (vLLM) and distributed training.
* Demonstrates expertise in Linux, HPC, networking, DevOps, and systems engineering.

## Summary

This project mirrors production HPC systems on a smaller scale, delivering cloud-like capabilities with local infrastructure. It showcases hands-on systems design, cluster orchestration, and technical leadership, which are critical skills for infrastructure and platform engineering roles.

## Links
[Architecture](https://github.com/CornellDataScience/computecluster/blob/main/architecture.md) \
[Adding a new node](https://github.com/CornellDataScience/computecluster/blob/main/newnode.md)
