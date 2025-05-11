# demo/image_demo.py (或者您复制修改后的版本)

import argparse
import os
import os.path as osp

import mmcv
import mmengine
import torch
from mmengine.structures import InstanceData

from mmdet.apis import init_detector
# For MMDetection 3.x, inference_detector is available but we'll call model parts directly
# from mmdet.apis import inference_detector # We will not use this directly for proposals
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample

# Import for data pipeline
# from mmdet.apis.utils import get_test_pipeline_cfg # Helper to get test pipeline
from mmcv.transforms import Compose # For MMDetection 2.x style pipeline composition if needed. MMDetection 3.x uses mmengine.dataset.Compose
# from mmengine.dataset.base_dataset import Compose # For MMDetection 3.x
# from mmcv.parallel import collate, scatter # Older MMDetection
from mmengine.dataset import pseudo_collate # For MMDetection 3.x for single sample

# Specific head for Sparse R-CNN if you want to type check (optional)
# from mmdet.models.dense_heads import SparseRCNNRPNHead


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection image demo')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./infer_output', help='Directory to save output visualizaiton file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold for final detections')
    parser.add_argument(
        '--show', action='store_true', help='Show the image')
    parser.add_argument(
        '--wait-time', type=float, default=0, help='The interval of show (s)')
    args = parser.parse_args()
    return args


def main(args):
    # Ensure output directory exists
    if args.out_dir:
        mmengine.mkdir_or_exist(args.out_dir)

    # Build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # Init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # The visualizer automatically uses dataset_meta from the model if available
    # Handle custom palette
    if args.palette == 'none':
        visualizer.dataset_meta = {}  # No palette
    elif args.palette: # For 'coco', 'voc', etc. MMDetection 3.x handles this via dataset_meta
        # For MMDetection 3.x, this is typically set via model.dataset_meta
        # If you need to override, you might set visualizer.dataset_meta['palette'] = args.palette
        # or ensure your config's visualizer_cfg is set up.
        # For simplicity, we assume model.dataset_meta or default palette works.
        # If model.dataset_meta is None or you want to force it:
        if not hasattr(model, 'dataset_meta') or model.dataset_meta is None:
            model.dataset_meta = {'classes': ('object',)} # Dummy classes if needed
        model.dataset_meta['palette'] = args.palette
        visualizer.dataset_meta = model.dataset_meta


    # Load image
    img_input_path = args.img
    img_orig = mmcv.imread(img_input_path) # For visualization later

    # === Step 1: Prepare data for the model ===
    # This mimics the data loading and preprocessing pipeline for a single image.
    cfg = model.cfg
    
    # Build the data pipeline
    # In MMDetection 3.x, test_pipeline is often part of test_dataloader
    if cfg.get('test_dataloader', {}).get('dataset', {}).get('pipeline'):
        test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
    else: # Fallback for older config structures or direct pipeline definition
        test_pipeline_cfg = get_test_pipeline_cfg(cfg)


    # Remove LoadAnnotations if it exists, as we are doing inference
    test_pipeline_cfg = [p for p in test_pipeline_cfg if p['type'] != 'LoadAnnotations']
    # Ensure LoadImageFromFile is the first step if img is a path
    if test_pipeline_cfg[0]['type'] != 'LoadImageFromFile':
        test_pipeline_cfg.insert(0, dict(type='LoadImageFromFile'))
    
    # MMDetection 3.x uses mmengine.dataset.Compose
    from mmengine.dataset.base_dataset import Compose as MMEngineCompose
    test_pipeline = MMEngineCompose(test_pipeline_cfg)

    # Prepare data dictionary
    data_info = {'img_path': img_input_path, 'img': img_orig.copy()} # Pass both path and loaded image if pipeline needs
    # Some pipelines might expect just 'img_path' and load it themselves.
    # Others might expect 'img' if 'LoadImageFromFile' is not the first.
    # To be safe, if LoadImageFromFile is present, provide img_path.
    # If it's already loaded and pipeline expects 'img', then that.
    # The above `data_info` should work if `LoadImageFromFile` handles `img_path`
    # and subsequent transforms use the loaded `img` field.

    data = test_pipeline(data_info) # Apply transformations

    # Collate and move to device using model.data_preprocessor
    # `data_preprocessor` expects a list of data dicts.
    processed_data = model.data_preprocessor([data], training=False)
    batch_inputs = processed_data['inputs'] # Tensor inputs
    data_samples_list = processed_data['data_samples'] # List of DetDataSample

    # === Step 2: Extract features ===
    with torch.no_grad():
        x = model.extract_feat(batch_inputs) # Backbone features

        # === Step 3: Get proposals from RPN head (or equivalent for Sparse R-CNN) ===
        # For Sparse R-CNN, rpn_head is usually SparseRCNNRPNHead (or similar name)
        # Its `predict` method gives initial proposals.
        proposals_visualized = False
        if hasattr(model, 'rpn_head') and hasattr(model.rpn_head, 'predict'):
            print("Attempting to get proposals from model.rpn_head.predict()...")
            try:
                # The `predict` method signature is typically (self, x, data_samples, **kwargs)
                rpn_outs = model.rpn_head.predict(x, data_samples_list)
                
                # For SparseRCNNRPNHead, rpn_outs is often a tuple:
                # (init_proposals_bboxes_list, init_proposals_features_list)
                # Each element in the tuple is a list (batch_size) of tensors.
                init_proposal_bboxes_list = rpn_outs[0]
                proposal_bboxes_tensor = init_proposal_bboxes_list[0].detach().cpu() # Get proposals for the first image

                # These proposals might be in normalized format (cx, cy, w, h) or (x1, y1, x2, y2).
                # The visualizer expects absolute coordinates (x1, y1, x2, y2).
                # Ensure they are in the correct format and scale.
                # If they are normalized cxcywh, convert them. Example (if needed):
                # metainfo = data_samples_list[0].metainfo
                # img_h, img_w = metainfo['img_shape'][:2] # Processed image shape
                # proposal_bboxes_tensor = bbox_cxcywh_to_xyxy(proposal_bboxes_tensor) # Assuming you have this utility
                # proposal_bboxes_tensor[:, 0::2] *= img_w
                # proposal_bboxes_tensor[:, 1::2] *= img_h
                # For Sparse R-CNN, they are usually already in xyxy and scaled to padded image size.
                # We will assume proposal_bboxes_tensor is in xyxy format.

                # === Step 4: Visualize proposals ===
                vis_data_sample_proposals = DetDataSample()
                # Use metainfo from the processed data_sample for correct coordinate context
                vis_data_sample_proposals.set_metainfo(data_samples_list[0].metainfo)

                pred_instances_proposals = InstanceData()
                pred_instances_proposals.bboxes = proposal_bboxes_tensor
                
                if proposal_bboxes_tensor.numel() > 0:
                    # Add dummy scores and labels for visualization
                    pred_instances_proposals.scores = torch.ones(proposal_bboxes_tensor.size(0))
                    pred_instances_proposals.labels = torch.zeros(proposal_bboxes_tensor.size(0), dtype=torch.long)
                
                vis_data_sample_proposals.pred_instances = pred_instances_proposals

                proposal_out_file = None
                if args.out_dir:
                    img_filename = osp.basename(img_input_path)
                    base, ext = osp.splitext(img_filename)
                    proposal_out_file = osp.join(args.out_dir, f'{base}_proposals{ext}')

                visualizer.add_datasample(
                    name='proposals',
                    image=img_orig.copy(), # Use the original image for visualization
                    data_sample=vis_data_sample_proposals,
                    draw_gt=False,
                    draw_pred=True,
                    wait_time=args.wait_time,
                    out_file=proposal_out_file,
                    pred_score_thr=0.0 # Show all initial proposals
                )
                proposals_visualized = True
                if proposal_out_file:
                    print(f"Proposal visualization saved to {proposal_out_file}")
                if args.show:
                    visualizer.show() # Show proposals if requested

            except Exception as e:
                print(f"Could not get or visualize proposals: {e}")
                import traceback
                traceback.print_exc()

        if not proposals_visualized:
            print("Could not find or visualize proposals from rpn_head. Falling back to final detections.")

        # === Visualize final detections as well (optional, for comparison) ===
        print("Visualizing final detections...")
        # Use the standard model.predict for final results
        final_results_list = model.predict([{'img_path': img_input_path, 'img': img_orig.copy()}]) # Re-run predict for final
        final_result_datasample = final_results_list[0]

        final_out_file = None
        if args.out_dir:
            img_filename = osp.basename(img_input_path)
            base, ext = osp.splitext(img_filename)
            final_out_file = osp.join(args.out_dir, f'{base}_final_detections{ext}')
        
        visualizer.add_datasample(
            name='final_detections',
            image=img_orig.copy(),
            data_sample=final_result_datasample,
            draw_gt=False,
            draw_pred=True,
            wait_time=args.wait_time, # Show if specified
            out_file=final_out_file,
            pred_score_thr=args.score_thr
        )
        if final_out_file:
            print(f"Final detection visualization saved to {final_out_file}")
        if args.show and not proposals_visualized: # Only show again if proposals weren't shown
             visualizer.show()
        elif args.show and proposals_visualized: # If proposals were shown, this might open a new window or update
            print("Final detections visualized. May require closing previous window if not interactive.")


if __name__ == '__main__':
    args = parse_args()
    main(args)