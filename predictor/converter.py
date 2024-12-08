from nuscenes import NuScenes
from itertools import permutations
import pandas as pd
import logging
import numpy as np
import torch
import json
from .hivt.utils import TemporalData
import tqdm

HISTORICAL_FRAMES = 4
PREDICT_FRAMES = 6
FRAMES = 10
EGO_DIST = 20

CLASSES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'barrier']

def get_forecast_data(version, data_root, track_results, save=True, save_path='trajectory.json'):
    nusc = NuScenes(version=version, dataroot=data_root)
    track_lst = []
    for seq_id, tracks in track_results['results'].items():
        sample = nusc.get('sample', seq_id)
        scene_token = sample['scene_token']
        for info in tracks:
            row = {
                'scene_token': scene_token,
                'sample_token': info['sample_token'],
                'timestamp': sample['timestamp'],
                'transform': info['translation'],  
                'bbox': info['size'] + info['rotation'],
                'label': CLASSES.index(info['tracking_name']),
                'name': info['tracking_name'],
                'velocity': info['velocity'],
                'tracking_id': info['tracking_id'],
                'tracking_score': info['tracking_score']
            }
            track_lst.append(row)
    
    track_results_df =  pd.DataFrame(track_lst)
    trajs_dict = process_argoverse_v2(track_results_df)

    torch.save(trajs_dict, save_path)
    trajs_dict = torch.load(save_path)
    graph_data = []
    for seq_id, value in trajs_dict.items():
        graph_data.append(TemporalData(**value))
    return graph_data

def process_argoverse_v2(raw_df: pd.DataFrame,
                         split: str='val'):
    scene_tokens = list(raw_df['scene_token'].unique())
    logging.info(f"Current data has {len(scene_tokens)} scenes!")
    raw_df_lst = [raw_df[raw_df['scene_token'] == scene_token] for scene_token in scene_tokens]
    timestamps_all = []
    fragments_all = []
    # convert tracking results to trajectory fragments
    for raw_df_single in raw_df_lst:
        sample_tokens = list(raw_df_single['sample_token'].unique())
        timestamps = list(np.sort(raw_df_single['timestamp'].unique()))
        timestamps_lst = [timestamps[i:i+FRAMES] for i in range(0, len(timestamps)-FRAMES-1)]
        fragment_df_lst = [raw_df_single[raw_df_single['timestamp'].isin(ts)] for ts in timestamps_lst]
        timestamps_all.extend(timestamps_lst)
        fragments_all.extend(fragment_df_lst)
    
    info_lst = {}
    
    i = 0

    for timestamps, fragment_df in tqdm.tqdm(zip(timestamps_all, fragments_all)):
        historical_timestamps = timestamps[:HISTORICAL_FRAMES]
        historical_df = fragment_df[fragment_df['timestamp'].isin(historical_timestamps)]
        actor_ids = list(historical_df['tracking_id'].unique())
        fragment_df = fragment_df[fragment_df['tracking_id'].isin(actor_ids)]
        
        for actor_id, actor_df in fragment_df.groupby('tracking_id'):
            # ignore the object if the object's historical frames are less than 5
            node_steps_observed = get_agent_node_steps(actor_df, timestamps, 'observed')
            if HISTORICAL_FRAMES -1 not in node_steps_observed or len(node_steps_observed) < 2:
                fragment_df = fragment_df[fragment_df['tracking_id'] != actor_id]
                actor_ids.remove(actor_id)
                continue
        num_nodes = len(actor_ids)

        if num_nodes == 0 or num_nodes == 1:
            print("\nCouldn't find complete trajectories!")
            continue
        
        for agent_id in actor_ids:
            av_df = fragment_df[fragment_df['tracking_id'] == agent_id]
            av_df_iloc = av_df.iloc
            steps = len(get_agent_node_steps(av_df, timestamps, 'observed'))
            av_index = actor_ids.index(av_df_iloc[0]['tracking_id'])
            actors_id = []
            # agent_df = fragment_df[fragment_df['gt_names'] == 'PEDESTRIAN'].iloc
            # agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])

            origin = torch.tensor([av_df_iloc[steps-1]['transform'][0], av_df_iloc[steps-1]['transform'][1]], dtype=torch.float)
            av_heading_vector = origin - torch.tensor([av_df_iloc[steps-2]['transform'][0], av_df_iloc[steps-2]['transform'][1]], dtype=torch.float)
            theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
            rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                    [torch.sin(theta), torch.cos(theta)]])

            # initialization
            x = torch.zeros(num_nodes, FRAMES, 2, dtype=torch.float)
            edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous() # 生成fully-connected边
            padding_mask = torch.ones(num_nodes, FRAMES, dtype=torch.bool)
            bos_mask = torch.zeros(num_nodes, FRAMES, dtype=torch.bool)
            rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
            
            # boxes_2d = torch.zeros(num_nodes, 20, 4, dtype=torch.int32)
            
            for actor_id, actor_df in fragment_df.groupby('tracking_id'):
                node_idx = actor_ids.index(actor_id)
                node_steps = [timestamps.index(timestamp) for timestamp in actor_df['timestamp']]
                padding_mask[node_idx, node_steps] = False
                if padding_mask[node_idx, HISTORICAL_FRAMES-1]:  # make no predictions for actors that are unseen at the current time step
                    padding_mask[node_idx, HISTORICAL_FRAMES:] = True
                xy = torch.from_numpy(np.stack([np.array(list(actor_df['transform']))[:, 0], np.array(list(actor_df['transform']))[:, 1]], axis=-1)).float()
                x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
                node_historical_steps = list(filter(lambda node_step: node_step < HISTORICAL_FRAMES, node_steps))
                if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
                    heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
                    rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
                else:  # make no predictions for the actor if the number of valid time steps is less than 2
                    padding_mask[node_idx, HISTORICAL_FRAMES:] = True
                actors_id.append(actor_id)
            bos_mask[:, 0] = ~padding_mask[:, 0]
            bos_mask[:, 1: HISTORICAL_FRAMES] = padding_mask[:, : HISTORICAL_FRAMES-1] & ~padding_mask[:, 1: HISTORICAL_FRAMES]

            positions = x.clone()
            x[:, HISTORICAL_FRAMES:] = torch.where((padding_mask[:, HISTORICAL_FRAMES-1].unsqueeze(-1) | padding_mask[:, HISTORICAL_FRAMES:]).unsqueeze(-1),
                                    torch.zeros(num_nodes, PREDICT_FRAMES, 2),
                                    x[:, HISTORICAL_FRAMES:] - x[:, HISTORICAL_FRAMES-1].unsqueeze(-2))
            x[:, 1: HISTORICAL_FRAMES] = torch.where((padding_mask[:, : HISTORICAL_FRAMES-1] | padding_mask[:, 1: HISTORICAL_FRAMES]).unsqueeze(-1),
                                    torch.zeros(num_nodes, HISTORICAL_FRAMES-1, 2),
                                    x[:, 1: HISTORICAL_FRAMES] - x[:, : HISTORICAL_FRAMES-1])
            x[:, 0] = torch.zeros(num_nodes, 2)

            y = None if split == 'test' else x[:, HISTORICAL_FRAMES:]

            info_lst[i] = {
                'x': x[:, : HISTORICAL_FRAMES],  # [N, 4, 2]
                'positions': positions,  # [N, 10, 2]
                'edge_index': edge_index,  # [2, N x N - 1]
                'y': y,  # [N, 6, 2]
                'num_nodes': num_nodes,
                'padding_mask': padding_mask,  # [N, 10]
                'bos_mask': bos_mask,  # [N, 4]
                'rotate_angles': rotate_angles,  # [N]
                'av_index': av_index,
                'origin': origin.unsqueeze(0),
                'theta': theta,
                'actors_id': np.array(actor_ids)
            }
            
            i += 1

    return info_lst

def get_center_agent(df, agent_type):
    log_ids = list(df['gt_uuid'].unique())
    agent_ids = []
    for log_id in log_ids:
        agent_df = df[df['gt_uuid'] == log_id]
        if agent_df['gt_name'].iloc[0] == agent_type:
            agent_ids.append(log_id)
    return agent_ids

def get_agent_node_steps(df, timestamps, type = 'all'):
    node_steps = [timestamps.index(timestamp) for timestamp in df['timestamp']]
    if type == 'all':
        return node_steps
    elif type == 'observed':
        observed_node_steps = [step for step in node_steps if step < HISTORICAL_FRAMES]
        return observed_node_steps
    elif type == 'predict':
        predict_node_steps = [step for step in node_steps if step >= HISTORICAL_FRAMES]
        return predict_node_steps