# reachy2_stack/perception/localization/hloc_mapping.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R  
import math
import cv2
import json

from reachy2_stack.utils.utils_mapping import process_leica_for_hloc, MappingInputLeica, HLocMapConfig
from reachy2_stack.perception.localization.hloc_localizer import HLocLocalizer



def build_hloc_map_from_leica(
    mapping_input: MappingInputLeica,
    cfg: HLocMapConfig,
) -> Path:
    """Build or update an HLoc/ COLMAP map for one location.
    build .mat files from rendered RGBD panoramas and poses and bring into hloc format

    Returns:
        Path to the created map directory, e.g. maps_root/location_name
    """

    # process leica data for hloc
    # leica mesh and pano are converted into .mat files which hloc inloc consumes
    # saved to maps_root/location_name_processed_leica
    leica_out_base = cfg.maps_root / f"{cfg.location_name}_processed_leica"
    process_leica_for_hloc(pano_path=mapping_input.pano_path,
                           pano_pose_path=mapping_input.pano_pose_path,
                           mesh_path=mapping_input.mesh_path,
                           out_base=leica_out_base,
                           overwrite=cfg.overwrite
                           )
    
    hloc_localizer = HLocLocalizer(
        cfg=cfg
    )
    hloc_localizer.setup_database_structure()
    
    a = 2

  