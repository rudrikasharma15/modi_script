# Handwritten Sample Testing Report

## Overview
- **Model**: runs/modi_matra/train_full_7k2/weights/best.pt
- **Test Images**: 61
- **Success Rate**: 77.0%

## Results Summary

### Detection Statistics
- ✅ Successful: 47 (77.0%)
- ❌ Failed: 14 (23.0%)

### By Matra Class
| Class | Detections | Avg Confidence |
|-------|-----------|----------------|
| bottom_matra | 11 | 0.822 |
| side_matra | 10 | 0.440 |
| top_matra | 42 | 0.814 |

### By Character Size
| Size | Detections |
|------|-----------|
| unknown | 63 |

## Verdict
**GOOD** - 77.0% success rate

## Files
- Annotated images: `handwritten_test_results/`
- Detailed JSON report: `handwritten_test_report.json`
