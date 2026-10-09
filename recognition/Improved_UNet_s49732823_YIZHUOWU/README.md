# 3D Improved U-Net for Prostate Segmentation (HipMRI)

**Student Name:** Yizhuo Wu  
**Student ID:** 49732823  
**Course:** COMP3710 / COMP7710 - Pattern Recognition (2026)  
**Track:** Domain 2.2 Segmentation — Anatomical Contouring and Radiotherapy Planning  
**Target Model:** 3D Improved U-Net  (Hard Difficulty)
**Baseline Model:** 2D U-Net

## 1. Project Description
This project investigates automated anatomical contouring of the prostate gland from pelvic MRI scans (HipMRI dataset) for targeted radiotherapy planning. The objective is to evaluate whether 3D volumetric context (3D Improved U-Net) significantly resolves slice-transition artifacts and clinical boundary leakage (into adjacent rectal wall and bladder mucosa) compared to a slice-wise 2D U-Net baseline, under strict patient-level separation.

## 2. Data & Setup
- **Dataset:** HipMRI Study (`/home/groups/comp3710/HipMRI_Study_open`)
- **Framework:** PyTorch