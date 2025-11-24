---
layout: post
title: Creating 3D Topographic Maps with Blender and QGIS 
categories: [geospatial]
date: "2025-11-23"
author: "Ferdinand Schenck"
format: 
  html:
    toc: true
    embed-resources: true
    code-fold: true
description: Done as part of the 30DayMapChallenge 2025.
---

DEM From
https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3


"Raster->Extraction->Contour"

Setting pic 100

Copper: #B87333


Project->Layout Manager...

Add Item->Add Map

Click on top left of canvas and draw map


Layout->Export as Image


Blender

In Layout mode. Add->Mesh->Plane

Edit Mode->Right click on plane. Subdivide. 

Hebed recommends 100 cuts, but I found that too coarse, so try 1000, if your computer can handle it. 

back to object mode

Modifier properties (little blue wrench) 

Add modifier-> Deform -> Displace
Add new texture

Texture properties

Open -> Choose your DEM PNG

Add Modifier -> Generate -> Subdivision Surface 

Object -> Scale Z around 20

Material property 

Add new color map

Surface -> Base Color -> Image Texture

Select the light -> Light. 

Make it a Sun, and put strength to 3 or 4

Set the angle. Based on what you do, you might want a higher or lower angle. I went for 30 here


Camera Orthographic projection

