# ---------------------------------------------------------------------------------------------------------------
line_color_dict = {'yellow': 0, 'white': 1, 'unknown': 2}

line_type_dict = {'double_solid': 0, 'single_solid': 1, 'right_solid_left_dashed': 2, 'single_dashed': 3,
                'left_solid_right_dashed': 4, 'double_dashed': 5, 'road_edge': 6, 'unknown': 7}

line_pos_dict = {'position_invalid': 0, 'nnext_right': 1, 'nnext_left': 2, 'host_right': 3, 'host_left': 4, 
'next_left': 5, 'next_right': 6, 'curb_left': 7, 'curb_right': 8, 'unknown': 9}

line_para_items = ['coef_a', 'coef_b', 'coef_c', 'coef_d', 'line_start_x', 'line_end_x']
# ---------------------------------------------------------------------------------------------------------------
obs_type_dict = {'truck': 0, 'car': 1, 'cyclist': 2, 'unknown': 3, 'pedestrian': 4, 'cone': 5, 'bus': 6}
obs_para_items = ['geo_x', 'geo_y', 'geo_z', 'distance', 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'accel_x', 'accel_y', 'theta_angle']
# ---------------------------------------------------------------------------------------------------------------
ego_para_items = ['steer_angle','vel_x', 'vel_y', 'accel_x', 'accel_y', 'euler_yaw']
# ---------------------------------------------------------------------------------------------------------------
camera_name_2_postfix = {
                        'front_wide_camera_record': '_0',
                        'rf_wide_camera_record': '_1',
                        'rr_wide_camera_record': '_2',
                        'rear_wide_camera_record': '_3',
                        'lr_wide_camera_record': '_4',
                        'lf_wide_camera_record': '_5'
                        }
# ---------------------------------------------------------------------------------------------------------------
 # 1：左变道，2：右变道，3：直行（巡航），4：左掉头，5：路口左转，6：路口右转
ego_motions = {'左变道':1, '右变道':2, '左转':3, '右转':4, '掉头':5, '直行':6, '巡航':7}