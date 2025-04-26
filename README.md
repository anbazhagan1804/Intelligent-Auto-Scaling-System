# Intelligent Auto-Scaling System

## Overview
This project implements an intelligent auto-scaling system using Kubernetes, Prometheus, KEDA, Terraform, and Python. It leverages real-time monitoring and ML-based predictive scaling to optimize resource usage.

## Structure
- `infrastructure/`: Terraform scripts for provisioning cloud and Kubernetes resources.
- `monitoring/`: Prometheus and monitoring configurations.
- `scaling/`: KEDA and custom scaling logic.
- `ml/`: Python code for predictive scaling using ML models.

## Getting Started
1. Review each subdirectory for setup instructions.
2. Deploy infrastructure, monitoring, and scaling components.
3. Integrate the ML service for predictive scaling.

---

## Goals
- Real-time, intelligent scaling based on diverse metrics
- Predictive scaling using ML models
- Infrastructure-as-code for reproducibility