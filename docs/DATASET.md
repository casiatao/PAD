## MSIP dataset

Download the eight publicly available sub-datasets firstly, and then follow the steps  below to extract images.

- DroneVehicle:  Extract all 17,990 infrared training images and use the script `/pretraining/dataset/extract/drone_vehicle_extract.py` to remove the white boarders from the images.
- SCUT FIR Pedestrian: Use the script `/pretraining/dataset/extract/scit_fir_pedestrian_extract.py` to extract 21,049 images.
- LLVIP: Extract all 12,025 infrared training images.
- Infrared Aerial Photography: Use the script `/pretraining/dataset/extract/infra_aerial_extract.py` to extract 5,523 images.
- Infrared Ship: Extract all 8,402 infrared training images.
- Infrared Security: Extract all 8,999 infrared images.
- LSOTB-TIR: Use the script `/pretraining/dataset/extract/lsotb_tir_extract.py` to extract 52,925 infrared training images.
- LasHeR: Use the script `/pretraining/dataset/extract/lasher_extract.py` to extract 51,843 infrared training images.

The MSIP dataset should be organized in the following format:

```
MSIP/
	dronevehicle/
		thermal/
			image1.jpg
			image2.jpg
			...
	scut_fir_pedestrian/
		thermal/
			image1.jpg
			image2.jpg
			...
	...
```

