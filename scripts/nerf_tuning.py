import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# loop over paramters and run the script
TRAINSET_SIZE = 34 # need to manually adjust in provider.py
MAX_EPOCH = 150
dt_gammas = [0, 1/256]
schedulers = ['LambdaLR', 'ReduceLROnPlateau']
reduce_lr_patiences = [5, 10]
reduce_lr_factors = [0.9]
error_maps = [True, False]
error_map_weights = [0.1, 0.01]

for dt_gamma in dt_gammas:
    for scheduler in schedulers:
        for reduce_lr_patience in reduce_lr_patiences:
            for reduce_lr_factor in reduce_lr_factors:
                for error_map in error_maps:
                    for error_map_weight in error_map_weights:

                        workspace_name = f"ngp-dtgamma{dt_gamma}-{scheduler}-trainset_size{TRAINSET_SIZE}"
                        if scheduler == 'ReduceLROnPlateau':
                            workspace_name += f"-patience{reduce_lr_patience}-redlrfactor{reduce_lr_factor}"
                        if error_map:
                            workspace_name += "-errmap"
                            workspace_name += f"-errmap_weight{error_map_weight}"

                        cmd = f"CUDA_VISIBLE_DEVICES=0 python main_nerf.py nerf_synthetic/lego --workspace {workspace_name} -O --tcnn --bound 1.5 --scale 1 --mode blender --dt_gamma {dt_gamma} --scheduler {scheduler} --max_epoch {MAX_EPOCH} -w"
                        if scheduler == 'ReduceLROnPlateau':
                            cmd += f" --reduce_lr_patience {reduce_lr_patience} --reduce_lr_factor {reduce_lr_factor}"
                        if error_map:
                            cmd += f" --error_map --error_map_weight {error_map_weight}"

                        # skip duplicated error_map_weights if error_map is False
                        if not error_map:
                            if error_map_weight in error_map_weights[1:]:
                                continue
                        # skip duplicated reduce_lr_factors etc if scheduler is not ReduceLROnPlateau
                        if scheduler == 'LambdaLR':
                            if reduce_lr_patience in reduce_lr_patiences[1:]:
                                continue
                            if reduce_lr_factor in reduce_lr_factors[1:]:
                                continue
                        print(cmd)
                        os.system(cmd)
