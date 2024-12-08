from nuscenes import NuScenes
import logging
import os

def converter(
                data_root,
                version='v1.0-mini', 
                results=None,
                result_path=None
            ):
    nusc = NuScenes(version, data_root)
    if results is None:
        assert os.path.exists(result_path), print("Please input correct results path!")

    results = results['results']
    for sample_id, track_lst in results.items():
        sample_id, 