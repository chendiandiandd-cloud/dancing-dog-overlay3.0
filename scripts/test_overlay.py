#!/usr/bin/env python3
"""Black-box regression checks using bundled dogs and generated input fixtures."""
import argparse
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def invoke(args, expect_ok=True):
    p = subprocess.run([sys.executable, str(ROOT/'scripts/render_overlay.py'), *args],
                       capture_output=True, text=True, timeout=180)
    if expect_ok and p.returncode:
        raise AssertionError(f'CLI failed ({p.returncode}): {p.stdout[-3000:]} {p.stderr[-3000:]}')
    return p


def probe(path):
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                        '-of', 'json', str(path)], capture_output=True, text=True, check=True, timeout=30)
    return json.loads(p.stdout)


def plans(directory):
    found = []
    def walk(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get('shots'), list) and obj['shots']:
                found.append(obj)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    for p in Path(directory).rglob('*.json'):
        try:
            walk(json.loads(p.read_text()))
        except (ValueError, UnicodeError):
            continue
    if not found:
        raise AssertionError(f'No machine-readable shot plan under {directory}')
    return found


class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'temp_data').mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix='dog-overlay-tests-', dir=ROOT/'temp_data')
        cls.base = Path(cls.temp.name)
        cls.fixtures = cls.base/'fixtures'
        subprocess.run([sys.executable, str(ROOT/'scripts/make_test_inputs.py'), '--output-dir', str(cls.fixtures)],
                       capture_output=True, text=True, check=True, timeout=120)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def out(self, name):
        d = self.base/name
        d.mkdir(exist_ok=True)
        return d

    def test_01_asset_integrity(self):
        manifest = json.loads((ROOT/'assets/dog_manifest.json').read_text())
        self.assertEqual(len(manifest['dogs']), 16)
        self.assertEqual(len({d['id'] for d in manifest['dogs']}), 16)
        for d in manifest['dogs']:
            with np.load(ROOT/'assets'/d['file'], allow_pickle=False) as z:
                f = z['frames']
                self.assertEqual(f.dtype, np.uint8)
                self.assertEqual(f.shape, (d['frame_count'], d['height'], d['width'], 4))
                self.assertGreater(np.count_nonzero(f[...,3]), 100)
                self.assertGreater(np.count_nonzero(f[...,3] == 0), 100)
                self.assertTrue(0 <= d['anchor'][0] <= d['width'])
                self.assertTrue(0 <= d['anchor'][1] <= d['height'])

    def test_02_help(self):
        self.assertIn('--inputs', invoke(['--help']).stdout)

    def test_03_manual_cuts_and_nonrepeat(self):
        out = self.out('cuts')
        invoke(['--inputs', str(self.fixtures/'three_shots.mp4'), '--output-dir', str(out),
                '--cuts', '1,2', '--seed', '7', '--plan-only'])
        shot_plan = plans(out)[0]['shots']
        self.assertEqual(len(shot_plan), 3)
        ids = [s['dog_id'] for s in shot_plan]
        self.assertTrue(all(a != b for a,b in zip(ids, ids[1:])))
        self.assertAlmostEqual(shot_plan[1]['start'], 1, places=2)
        self.assertAlmostEqual(shot_plan[2]['start'], 2, places=2)

    def test_04_seed_reproducibility(self):
        results=[]
        for name in ('seed_a','seed_b'):
            out=self.out(name)
            invoke(['--inputs', str(self.fixtures/'three_shots.mp4'), '--output-dir', str(out),
                    '--cuts', '1,2', '--seed', '42', '--plan-only'])
            # Reproducibility concerns rendered dog/geometry/time, not tiny
            # parallel floating-point differences in diagnostic texture scores.
            shots=plans(out)[0]['shots']
            results.append([{
                'start':s['start'], 'end':s['end'], 'dog_id':s['dog_id'],
                'phase_frames':s['phase_frames'],
                'anchor':[round(x,6) for x in s['anchor']],
                'height_ratio':round(s['height_ratio'],6),
            } for s in shots])
        self.assertEqual(results[0], results[1])

    def test_05_photo_movie(self):
        out=self.out('photo')
        invoke(['--inputs', str(self.fixtures/'scene_1.png'), '--output-dir', str(out),
                '--image-duration', '0.5', '--seed', '1'])
        mp4=list(out.rglob('*.mp4'))
        self.assertEqual(len(mp4),1)
        d=probe(mp4[0]); v=next(s for s in d['streams'] if s['codec_type']=='video')
        self.assertEqual((v['width'],v['height']),(960,540))
        self.assertLess(abs(float(d['format']['duration'])-0.5),0.06)
        self.assertFalse(any(s['codec_type']=='audio' for s in d['streams']))

    def test_06_video_audio_duration(self):
        out=self.out('audio')
        invoke(['--inputs', str(self.fixtures/'three_shots.mp4'), '--output-dir', str(out),
                '--cuts', '1,2', '--seed','2'])
        d=probe(next(out.rglob('*.mp4')))
        self.assertTrue(any(s['codec_type']=='audio' for s in d['streams']))
        self.assertLess(abs(float(d['format']['duration'])-3),0.06)

    def test_07_portrait_silent(self):
        out=self.out('portrait')
        invoke(['--inputs',str(self.fixtures/'portrait_silent.mp4'),'--output-dir',str(out),'--seed','3'])
        d=probe(next(out.rglob('*.mp4')))
        v=next(s for s in d['streams'] if s['codec_type']=='video')
        self.assertEqual((v['width'],v['height']),(360,640))
        self.assertFalse(any(s['codec_type']=='audio' for s in d['streams']))

    def test_08_mixed_failure_isolation(self):
        out=self.out('batch')
        p=invoke(['--inputs',str(self.fixtures/'scene_2.png'),str(self.fixtures/'broken.mp4'),
                  '--output-dir',str(out),'--image-duration','0.5'],expect_ok=False)
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(len(list(out.rglob('*.mp4'))),1)
        self.assertTrue(list(out.rglob('*.json')))

    def test_09_bad_input_fails(self):
        out=self.out('bad')
        p=invoke(['--inputs',str(self.fixtures/'broken.mp4'),'--output-dir',str(out)],expect_ok=False)
        self.assertNotEqual(p.returncode,0)
        self.assertFalse(list(out.rglob('*.mp4')))

    def test_10_automatic_cuts(self):
        out=self.out('automatic')
        invoke(['--inputs',str(self.fixtures/'three_shots.mp4'),'--output-dir',str(out),
                '--seed','5','--plan-only'])
        shots=plans(out)[0]['shots']
        self.assertEqual(len(shots),3)
        self.assertLess(abs(shots[1]['start']-1),0.09)
        self.assertLess(abs(shots[2]['start']-2),0.09)


    def mock_dog(self):
        frames = np.zeros((3, 100, 100, 4), np.uint8)
        frames[:, 3:96, 3:97] = [210, 80, 30, 255]
        return dict(frames=frames, frame_count=3, width=100, height=100,
                    anchor=(50, 96), union_bbox=[3, 3, 97, 96])

    def test_11_adaptive_joint_size_and_protection(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement, intersects
        info = dict(display_width=400, display_height=400)
        blank = np.full((400, 400, 3), 180, np.uint8)
        with patch('overlay_core.face_boxes', return_value=([], True)), patch('overlay_core.person_boxes', return_value=([], True)):
            large = placement(blank, info, self.mock_dog(), None, 'auto', random.Random(42))
            again = placement(blank, info, self.mock_dog(), None, 'auto', random.Random(42))
        self.assertEqual(large, again)
        self.assertGreaterEqual(large['actual_height_ratio'], .24)
        self.assertLessEqual(large['actual_height_ratio'], .40)
        self.assertGreater(large['anchor_normalized'][1], .6)
        self.assertEqual(len(large['evaluated_sizes']), 5)
        obstacle = [145, 0, 400, 400]
        with patch('overlay_core.face_boxes', return_value=([], True)), patch('overlay_core.person_boxes', return_value=([obstacle], True)):
            smaller = placement(blank, info, self.mock_dog(), None, 'auto', random.Random(42))
        self.assertLess(smaller['actual_height_ratio'], large['actual_height_ratio'])
        self.assertGreaterEqual(smaller['actual_height_ratio'], .24)
        self.assertLessEqual(smaller['width'], 145)
        self.assertFalse(intersects([smaller['x'], smaller['y'], smaller['x']+smaller['width'], smaller['y']+smaller['height']], obstacle))
        self.assertEqual(smaller['size_mode'], 'adaptive')

    def test_12_manual_override_and_boundary_ratio(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement, PreparedDog
        blank = np.full((400, 400, 3), 180, np.uint8)
        info = dict(display_width=400, display_height=400)
        with patch('overlay_core.face_boxes', return_value=([], True)), patch('overlay_core.person_boxes', return_value=([], True)):
            exact = placement(blank, info, self.mock_dog(), .12, 'auto', random.Random(3), [.5, .8])
            fitted = placement(blank, info, self.mock_dog(), .8, 'auto', random.Random(3), [.05, .15])
        self.assertEqual(exact['size_mode'], 'manual')
        self.assertAlmostEqual(exact['actual_height_ratio'], .12)
        self.assertLess(fitted['actual_height_ratio'], .8)
        self.assertEqual(fitted['actual_height_ratio'], fitted['height']/400)
        for p, anchor in [(exact, [.5, .8]), (fitted, [.05, .15])]:
            self.assertTrue(np.allclose(p['anchor_normalized'], anchor, atol=1/400))
            self.assertGreaterEqual(p['x'], 0); self.assertGreaterEqual(p['y'], 0)
            self.assertLessEqual(p['x']+p['width'], 400); self.assertLessEqual(p['y']+p['height'], 400)
            prep = PreparedDog(self.mock_dog(), p)
            for index in range(3):
                out = blank.copy(); prep.overlay(out, index)
                self.assertTrue(np.any(out != blank))

    def test_13_premultiplied_scaling_up_and_down(self):
        from overlay_core import PreparedDog
        from prepare_assets import resize_rgba
        rgba = np.zeros((20, 20, 4), np.uint8)
        rgba[..., :3] = [0, 255, 0]  # Invisible green must never contribute.
        rgba[5:15, 5:15] = [255, 60, 20, 255]
        for size in [(8, 8), (57, 57), (8, 57)]:
            result = resize_rgba(rgba, size)
            mask = result[..., 3] > 5
            self.assertTrue(np.any(mask))
            self.assertTrue(np.all(np.abs(result[mask, :3].astype(int)-[255,60,20]) <= 1))
            prep = PreparedDog(dict(frames=rgba[None], frame_count=1), dict(width=size[0], height=size[1]))
            color, alpha = prep.frame(0)
            self.assertTrue(np.all((alpha >= 0) & (alpha <= 1)))
            self.assertTrue(np.all((color >= 0) & (color <= 255*alpha+1e-4)))
            np.testing.assert_allclose(color[..., 1], color[..., 0]*60/255, atol=1e-4)
            np.testing.assert_allclose(color[..., 2], color[..., 0]*20/255, atol=1e-4)

    def test_14_native_asset_dimensions_and_source_accounting(self):
        from overlay_core import sha256, DogLibrary
        manifest = json.loads((ROOT/'assets/dog_manifest.json').read_text())
        total = 0
        for d in manifest['dogs']:
            p = d['resolution_provenance']
            x0,y0,x1,y1 = d['crop_bbox']
            self.assertEqual([d['width'],d['height']], [x1-x0,y1-y0])
            self.assertEqual(p['cache_resolution'], p['native_crop_resolution'])
            self.assertEqual(d['scale_xy'], [1.,1.])
            self.assertTrue(p['native_pixels_preserved'])
            self.assertEqual(p['sharpen_amount'], 0)
            self.assertEqual(sha256(ROOT/'assets'/d['source']), d['source_sha256'])
            self.assertEqual(sha256(ROOT/'assets'/d['file']), d['sha256'])
            total += d['frame_count']
            self.assertTrue(DogLibrary(ROOT/'assets').load(d['id'])['frame_count'] > 0)
        self.assertEqual(total, 827)
        d15 = next(d for d in manifest['dogs'] if d['id']=='dog_15')
        self.assertEqual(d15['runtime_source_frame_range'], [0,58])
        self.assertEqual(d15['trimmed_source_frame_indices'], [58])
        d06 = next(d for d in manifest['dogs'] if d['id']=='dog_06')
        self.assertTrue(d06['enclosed_green_policy']['preserve'])
        self.assertTrue(d06['quality']['protected_green_rgb_all_unchanged'])
        self.assertTrue(d06['quality']['protected_green_all_opaque'])
        self.assertGreater(d06['height'],384)

    def test_15_native_pixels_match_original_matte(self):
        import cv2
        from prepare_assets import matte
        manifest = json.loads((ROOT/'assets/dog_manifest.json').read_text())
        for dog_id in ['dog_06','dog_15']:
            d = next(d for d in manifest['dogs'] if d['id']==dog_id)
            cap = cv2.VideoCapture(str(ROOT/'assets'/d['source']))
            try:
                ok, bgr = cap.read(); self.assertTrue(ok)
            finally:
                cap.release()
            rgba, _ = matte(bgr, d['enclosed_green_policy']['preserve'])
            x0,y0,x1,y1 = d['crop_bbox']
            expected = np.zeros((y1-y0,x1-x0,4),np.uint8)
            sx0,sy0=max(0,x0),max(0,y0); sx1,sy1=min(rgba.shape[1],x1),min(rgba.shape[0],y1)
            expected[sy0-y0:sy1-y0,sx0-x0:sx1-x0] = rgba[sy0:sy1,sx0:sx1]
            with np.load(ROOT/'assets'/d['file'],allow_pickle=False) as z:
                np.testing.assert_array_equal(z['frames'][0], expected)

    def test_16_upper_structure_not_auto_support(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement
        scene = np.full((400,400,3),180,np.uint8)
        scene[100:115] = 25; scene[140:150] = 20
        with patch('overlay_core.face_boxes', return_value=([],True)), patch('overlay_core.person_boxes', return_value=([],True)):
            pos = placement(scene,dict(display_width=400,display_height=400),self.mock_dog(),None,'auto',random.Random(8))
        self.assertGreater(pos['anchor_normalized'][1], .6)
        self.assertNotEqual(pos['kind'], 'hough_horizontal_support')
        self.assertIn('selection_reason',pos)

    def test_17_local_person_region_geometry(self):
        from unittest.mock import patch
        from overlay_core import person_boxes
        with patch('overlay_core.cv2.HOGDescriptor_getDefaultPeopleDetector', None, create=True):
            # Small image skips HOG but still protects face-derived body extent.
            boxes, _ = person_boxes(np.zeros((100,50,3),np.uint8), [[15,20,25,30]])
        self.assertEqual(len(boxes),1)
        self.assertLess(boxes[0][0],15); self.assertGreater(boxes[0][3],30)
        self.assertTrue(0 <= boxes[0][0] < boxes[0][2] <= 50)


    def test_18_full_cycle_real_asset_bounds(self):
        import random
        from unittest.mock import patch
        from overlay_core import DogLibrary, placement, PreparedDog
        library = DogLibrary(ROOT/'assets')
        blank = np.full((640,360,3),180,np.uint8)
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            for dog_id in library.entries:
                dog = library.load(dog_id)
                p = placement(blank,dict(display_width=360,display_height=640),dog,.24,'auto',random.Random(8),[.4,.82])
                self.assertTrue(p['whole_canvas_bounds_checked'])
                prep = PreparedDog(dog,p)
                for i in range(dog['frame_count']):
                    color, alpha = prep.frame(i)
                    self.assertEqual(alpha.shape[:2],(p['height'],p['width']))
                    self.assertGreater(float(alpha.max()),0)
                    frame=blank.copy(); prep.overlay(frame,i)
                    delta=np.any(frame!=blank,axis=2)
                    outside=delta.copy(); outside[p['y']:p['y']+p['height'],p['x']:p['x']+p['width']]=False
                    self.assertFalse(np.any(outside))

    def test_19_manual_protection_conflict_fails(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement
        blank=np.full((200,200,3),180,np.uint8)
        with patch('overlay_core.face_boxes',return_value=([[50,50,150,150]],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            with self.assertRaises(ValueError):
                placement(blank,dict(display_width=200,display_height=200),self.mock_dog(),.24,'auto',random.Random(1),[.5,.5])

    def test_20_bundled_local_detectors_available(self):
        import cv2
        from overlay_core import face_boxes, person_boxes
        self.assertLess(int(cv2.__version__.split('.')[0]),5)
        rgb=np.zeros((256,128,3),np.uint8)
        _, haar=face_boxes(rgb)
        _, hog=person_boxes(rgb,[])
        self.assertTrue(haar); self.assertTrue(hog)

    def test_21_default_batch_photos_are_five_seconds(self):
        out = self.out('default_five_seconds')
        invoke(['--inputs', str(self.fixtures/'scene_1.png'),
                str(self.fixtures/'scene_2.png'), '--output-dir', str(out),
                '--seed', '1024', '--placement-mode', 'corner'])
        movies = list(out.rglob('*.mp4'))
        self.assertEqual(len(movies), 2)
        for movie in movies:
            data = probe(movie)
            self.assertLess(abs(float(data['format']['duration']) - 5.0), 0.06)
            self.assertFalse(any(s['codec_type'] == 'audio' for s in data['streams']))


    def motion_plan(self, mode='flow', cuts=(0.,)):
        return dict(tracking=mode, media=dict(display_width=320,display_height=240), shots=[
            dict(start=t, placement=dict(x=120,y=125,width=30,height=40,
                anchor_pixel=[135,165]), motion=dict(mode='flow',reason='lk_ransac_candidate',
                attempted_pairs=0,accepted_pairs=0,local_pairs=0,camera_pairs=0,
                confidence_sum=0.,rejection_reasons={},trajectory=[])) for t in cuts])

    def textured_frames(self, n=18, dx=2):
        import cv2
        rng=np.random.default_rng(72)
        im=rng.integers(0,256,(240,320,3),np.uint8)
        im=cv2.GaussianBlur(im,(3,3),0)
        return [cv2.warpAffine(im,np.float32([[1,0,k*dx],[0,1,0]]),(320,240),
                               borderMode=cv2.BORDER_REFLECT) for k in range(n)]

    def test_22_sparse_lk_real_translation_and_interpolation(self):
        from motion_tracking import follow_frames,tracking_offset
        plan=self.motion_plan(); frames=self.textured_frames()
        follow_frames(iter(frames),plan,6,320,240)
        m=plan['shots'][0]['motion']
        self.assertGreater(m['accepted_ratio'],.95)
        self.assertAlmostEqual(m['trajectory'][-1]['dx'],34,delta=1)
        self.assertAlmostEqual(m['trajectory'][-1]['dy'],0,delta=1)
        self.assertAlmostEqual(tracking_offset(m,.25)[0],3,delta=.5)
        from overlay_core import PreparedDog
        dog=self.mock_dog(); prep=PreparedDog(dog,plan['shots'][0]['placement'])
        blank=np.full((240,320,3),180,np.uint8)
        out=blank.copy(); prep.overlay(out,0,tracking_offset(m,17/6))
        self.assertTrue(np.any(out[125:165,154:184] != blank[125:165,154:184]))
        self.assertFalse(np.any(out[:,:150] != blank[:,:150]))

    def test_23_low_texture_freeze_no_reacquisition_jump(self):
        from motion_tracking import follow_frames
        frames=self.textured_frames(12)
        frames[5]=np.full((240,320,3),128,np.uint8)
        plan=self.motion_plan(); follow_frames(iter(frames),plan,6,320,240)
        m=plan['shots'][0]['motion']
        self.assertIn('freeze_at',m); self.assertTrue(m['rejection_reasons'])
        offsets=[p['dx'] for p in m['trajectory']]
        self.assertAlmostEqual(offsets[4],offsets[-1],places=5)
        low=self.motion_plan(); follow_frames(iter([frames[5]]*9),low,6,320,240)
        self.assertEqual(low['shots'][0]['motion']['accepted_pairs'],0)
        self.assertEqual(low['shots'][0]['motion']['max_displacement_pixels'],0)

    def test_24_cut_resets_and_bounds_never_escape(self):
        from motion_tracking import follow_frames,safe_offset
        plan=self.motion_plan(cuts=(0.,1.5))
        follow_frames(iter(self.textured_frames()),plan,6,320,240)
        self.assertEqual(plan['shots'][1]['motion']['trajectory'][0]['dx'],0)
        self.assertLess(plan['shots'][1]['motion']['trajectory'][-1]['dx'],20)
        edge=self.motion_plan(); edge['shots'][0]['placement']['x']=286
        follow_frames(iter(self.textured_frames()),edge,6,320,240)
        m=edge['shots'][0]['motion']; self.assertIn('boundary',m['rejection_reasons'])
        for p in m['trajectory']:
            self.assertTrue(safe_offset(edge['shots'][0]['placement'],[p['dx'],p['dy']],320,240)[0])

    def test_25_all_time_subject_protection_and_static_auto(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement,intersects
        from motion_tracking import follow_frames
        im=np.full((400,400,3),180,np.uint8)
        obstacle=[0,230,110,400]
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',side_effect=[([],True),([obstacle],True)]):
            p=placement(im,dict(display_width=400,display_height=400),self.mock_dog(),None,'auto',random.Random(9),extra_frames=[im.copy()])
        self.assertFalse(intersects([p['x'],p['y'],p['x']+p['width'],p['y']+p['height']],obstacle))
        self.assertEqual(p['protection_sample_count'],2)
        plan=self.motion_plan('auto'); frames=self.textured_frames(12)
        frames[5]=np.full_like(frames[5],128)
        follow_frames(iter(frames),plan,6,320,240)
        self.assertEqual(plan['shots'][0]['motion']['mode'],'off')
        self.assertEqual(plan['shots'][0]['motion']['max_displacement_pixels'],0)

    def test_26_classic_text_and_four_corner_reserves(self):
        import cv2,random
        from unittest.mock import patch
        from classical_vision import text_regions,reserved_regions
        from overlay_core import placement,intersects
        im=np.full((400,400,3),180,np.uint8)
        cv2.putText(im,'WATERMARK TEXT',(190,360),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1,cv2.LINE_AA)
        regions=text_regions(im); self.assertTrue(regions)
        self.assertTrue(any(intersects(b,[190,348,330,363]) for b in regions))
        with patch('overlay_core.face_boxes',return_value=([],True)),patch('overlay_core.person_boxes',return_value=([],True)):
            p=placement(im,dict(display_width=400,display_height=400),self.mock_dog(),None,'auto',random.Random(3))
        rect=[p['x'],p['y'],p['x']+p['width'],p['y']+p['height']]
        self.assertFalse(any(intersects(rect,b) for b in regions+reserved_regions(400,400)))
        self.assertFalse(p['contact_shadow'])

    def test_27_large_pet_like_face_expansion_does_not_seal_frame(self):
        import random
        from unittest.mock import patch
        from overlay_core import person_boxes,placement,intersects
        im=np.full((256,144,3),180,np.uint8); face=[29,76,98,145]
        with patch('overlay_core.cv2.HOGDescriptor_getDefaultPeopleDetector',None,create=True):
            people,_=person_boxes(im,[face])
        self.assertGreater(people[0][0],0); self.assertLess(people[0][2],144)
        with patch('overlay_core.face_boxes',return_value=([face],True)),patch('overlay_core.person_boxes',return_value=(people,True)):
            p=placement(im,dict(display_width=144,display_height=256),self.mock_dog(),.15,'auto',random.Random(9),[.15,.25])
            with self.assertRaises(ValueError):
                placement(im,dict(display_width=144,display_height=256),self.mock_dog(),.15,'auto',random.Random(9),[.45,.45])
        self.assertFalse(intersects([p['x'],p['y'],p['x']+p['width'],p['y']+p['height']],face))

    def test_28_local_garment_segmentation_and_long_body_extent(self):
        import cv2
        from unittest.mock import patch
        from overlay_core import person_boxes
        from classical_vision import garment_mask
        im=np.full((320,240,3),[40,100,50],np.uint8)
        cv2.rectangle(im,(100,40),(120,60),(200,150,110),-1)
        cv2.fillPoly(im,[np.int32([[102,61],[118,61],[156,244],[68,244]])],(230,230,230))
        mask=garment_mask(im,[[100,40,121,61]])
        self.assertGreater(float(mask[80:180,99:122].mean()),.8)
        self.assertLess(float(mask[:,0:35].mean()),.01)
        with patch('overlay_core.cv2.HOGDescriptor_getDefaultPeopleDetector',None,create=True):
            boxes,_=person_boxes(im,[[100,40,121,61]])
        self.assertGreater(boxes[0][3],230)
        self.assertLess(boxes[0][0],85); self.assertGreater(boxes[0][2],135)

    def test_29_water_like_fragmented_support_is_sticker(self):
        import random,cv2
        from unittest.mock import patch
        from overlay_core import placement
        im=np.full((400,400,3),[80,130,160],np.uint8)
        for y in range(230,390,9):
            for x in range(0,400,23):
                cv2.line(im,(x,y),(x+8,y),(150,180,200),1)
        with patch('overlay_core.face_boxes',return_value=([],True)),patch('overlay_core.person_boxes',return_value=([],True)):
            p=placement(im,dict(display_width=400,display_height=400),self.mock_dog(),None,'auto',random.Random(8))
        self.assertEqual(p['placement_mode'],'screen_sticker')
        self.assertFalse(p['ground_anchored']); self.assertFalse(p['contact_shadow'])

    def test_30_cli_flow_renders_real_follow_not_noop(self):
        out=self.out('flow_public')
        invoke(['--inputs',str(self.fixtures/'three_shots.mp4'),'--output-dir',str(out),
                '--tracking','flow','--cuts','1,2','--seed','2'])
        p=plans(out)[0]
        self.assertEqual(len(p['shots']),3)
        self.assertTrue(all(s['motion']['attempted_pairs']>0 for s in p['shots']))
        self.assertTrue(all(s['motion']['trajectory'][0]['dx']==0 for s in p['shots']))
        self.assertTrue(Path(p['output']).is_file())

    def test_31_large_cascade_requires_classical_eye_evidence(self):
        from unittest.mock import patch,MagicMock
        from overlay_core import face_boxes
        face=MagicMock(); face.empty.return_value=False
        face.detectMultiScale.return_value=np.array([[29,76,69,69]])
        eye=MagicMock(); eye.empty.return_value=False; eye.detectMultiScale.return_value=[]
        rgb=np.zeros((256,144,3),np.uint8)
        with patch('overlay_core.cv2.CascadeClassifier',side_effect=[face,eye]):
            boxes,_=face_boxes(rgb)
        self.assertEqual(boxes,[])
        eye.detectMultiScale.return_value=np.array([[12,15,10,10]])
        with patch('overlay_core.cv2.CascadeClassifier',side_effect=[face,eye]):
            boxes,_=face_boxes(rgb)
        self.assertEqual(boxes,[[29,76,98,145]])

    def test_32_real_media_functional_contract(self):
        if not getattr(self,'real_summary',None):
            self.skipTest('supply --real-summary to verify independently rendered real media')
        from overlay_core import sha256
        from motion_tracking import safe_offset
        summary=json.loads(Path(self.real_summary).read_text())
        self.assertEqual(summary['counts']['failed'],0)
        self.assertEqual(summary['counts']['requested'],8)
        for result in summary['results']:
            p=json.loads(Path(result['plan_path']).read_text())
            self.assertEqual(p['input_sha256'],sha256(p['input']))
            data=probe(p['output']); v=next(s for s in data['streams'] if s['codec_type']=='video')
            self.assertAlmostEqual(float(v['duration']),p['media']['duration'],delta=1/p['media']['fps']+.025)
            self.assertEqual(bool(p['media']['audio']),any(s['codec_type']=='audio' for s in data['streams']))
            self.assertEqual(p['seed'],42); self.assertEqual(p['media']['quality'],'hd')
            for shot in p['shots']:
                self.assertTrue(safe_offset(shot['placement'],[0,0],p['media']['display_width'],p['media']['display_height'])[0])
                for point in shot['motion']['trajectory']:
                    self.assertTrue(safe_offset(shot['placement'],[point['dx'],point['dy']],p['media']['display_width'],p['media']['display_height'])[0])
            self.assertTrue(all(a['dog_id']!=b['dog_id'] for a,b in zip(p['shots'],p['shots'][1:])))


    def test_33_auto_stickers_skip_decoder_and_report_unobserved(self):
        from unittest.mock import patch
        from motion_tracking import plan_tracking, tracking_offset
        for kind, requested, anchor in [('video','auto',None),('video','off',None),
                                         ('video','flow',[.5,.5]),('image','flow',None)]:
            plan=self.motion_plan(requested,cuts=(0.,1.5))
            plan.update(requested_anchor=anchor,warnings=[])
            plan['media'].update(kind=kind,fps=6.)
            for s in plan['shots']:
                s['placement']['placement_mode']='screen_sticker'
            with patch('motion_tracking.video_frames') as decoder, patch('motion_tracking.sparse_motion') as lk:
                plan_tracking('unused',plan,20)
                decoder.assert_not_called(); lk.assert_not_called()
            self.assertEqual(plan['tracking_skipped_reason'],'no_flow_eligible_shots')
            for s in plan['shots']:
                m=s['motion']
                self.assertTrue(m['skipped_reason'])
                self.assertEqual(m['attempted_pairs'],0)
                self.assertIsNone(m['accepted_ratio']); self.assertIsNone(m['mean_confidence'])
                self.assertEqual(tracking_offset(m,2.),(0.,0.))

    def test_34_mixed_auto_only_surface_computes_flow(self):
        from unittest.mock import patch
        from motion_tracking import plan_tracking
        from classical_vision import sparse_motion
        plan=self.motion_plan('auto',cuts=(0.,1.5))
        plan.update(requested_anchor=None,warnings=[])
        plan['media'].update(kind='video',fps=6.)
        for i,s in enumerate(plan['shots']):
            s['placement']['placement_mode']='screen_sticker' if i==0 else 'surface'
        with patch('motion_tracking.video_frames',return_value=iter(self.textured_frames())) as decoder:
            # iter(list) has no close; use a generator, as the real decoder does.
            decoder.return_value=(f for f in self.textured_frames())
            with patch('motion_tracking.sparse_motion',wraps=sparse_motion) as lk:
                plan_tracking('unused',plan,20)
                self.assertGreater(lk.call_count,0)
        a,b=[s['motion'] for s in plan['shots']]
        self.assertEqual(a['attempted_pairs'],0); self.assertIsNone(a['accepted_ratio'])
        self.assertEqual(a['skipped_reason'],'safe_screen_sticker')
        self.assertEqual(b['attempted_pairs'],8)
        self.assertGreater(b['accepted_ratio'],.95)
        self.assertGreater(b['max_displacement_pixels'],10)
        self.assertEqual(b['trajectory'][0]['dx'],0)


    def test_35_user_regions_are_hard_constraints_and_tracking_safe(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement, intersects
        from motion_tracking import safe_offset
        im=np.full((400,400,3),180,np.uint8)
        regions=[[0,.5,.5,1],[.75,.5,1,1]]
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            p=placement(im,dict(display_width=400,display_height=400),self.mock_dog(),None,'auto',random.Random(3),avoid_regions=regions)
        rect=[p['x'],p['y'],p['x']+p['width'],p['y']+p['height']]
        self.assertEqual(p['user_avoid_regions_normalized'],regions)
        self.assertFalse(any(intersects(rect,b) for b in p['user_avoid_regions_pixels']))
        pos=dict(x=200,y=40,width=30,height=40,user_avoid_regions_pixels=[[0,200,200,400]])
        self.assertTrue(safe_offset(pos,[0,0],400,400)[0])
        self.assertEqual(safe_offset(pos,[-50,200],400,400),(False,'user_avoid_region_conflict'))

    def test_36_cli_rejects_invalid_user_regions(self):
        from overlay_core import validate_avoid_regions
        invalid=[[0,0,0,1],[.8,0,.2,1],[0,.7,1,.3],[-.1,0,1,1],
                 [0,0,1.1,1],[0,0,float('nan'),1],[0,0,1,float('inf')]]
        for region in invalid:
            with self.subTest(region=region):
                with self.assertRaises(ValueError): validate_avoid_regions([region])
                r=invoke(['--self-test','--avoid-region',*[str(v) for v in region]],expect_ok=False)
                self.assertEqual(r.returncode,2); self.assertIn('--avoid-region',r.stderr)
        r=invoke(['--self-test','--avoid-region','0','0','1'],expect_ok=False)
        self.assertEqual(r.returncode,2)
        with self.assertRaises(ValueError): validate_avoid_regions([[0,0,1]])

    def test_37_empty_regions_preserve_default_and_faces_stay_protected(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement
        im=np.full((400,400,3),180,np.uint8)
        kwargs=dict(rgb=im,info=dict(display_width=400,display_height=400),dog=self.mock_dog(),ratio=None,mode='auto')
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            a=placement(**kwargs,rng=random.Random(7))
            b=placement(**kwargs,rng=random.Random(7),avoid_regions=[])
        self.assertEqual(a,b)
        with patch('overlay_core.face_boxes',return_value=([[170,170,230,230]],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            with self.assertRaises(ValueError):
                placement(im,kwargs['info'],self.mock_dog(),.18,'auto',random.Random(7),[.5,.5],avoid_regions=[[0,0,.1,.1]])
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            with self.assertRaises(ValueError):
                placement(im,kwargs['info'],self.mock_dog(),.18,'auto',random.Random(7),[.5,.5],avoid_regions=[[.3,.3,.7,.7]])

    def test_38_cli_user_region_provenance_and_repeated_regions(self):
        from overlay_core import intersects
        out=self.out('user_regions')
        invoke(['--inputs',str(self.fixtures/'three_shots.mp4'),'--output-dir',str(out),
                '--plan-only','--seed','42','--avoid-region','0','.5','.45','1',
                '--avoid-region','.8','.5','1','1'])
        p=plans(out)[0]
        self.assertEqual(p['placement_assistance'],'user_avoid_regions')
        self.assertEqual(p['requested_avoid_regions'],[[0.,.5,.45,1.],[.8,.5,1.,1.]])
        for s in p['shots']:
            pos=s['placement']; rect=[pos['x'],pos['y'],pos['x']+pos['width'],pos['y']+pos['height']]
            self.assertFalse(any(intersects(rect,b) for b in pos['user_avoid_regions_pixels']))


    def structural_scenes(self):
        import cv2
        a = np.full((180,320,3), 18, np.uint8)
        cv2.rectangle(a,(30,20),(125,159),(105,95,85),-1)
        cv2.circle(a,(78,50),20,(160,140,120),-1)
        b = np.fliplr(a).copy()  # Identical whole-frame RGB histogram.
        c = np.full_like(a,18)
        cv2.rectangle(c,(75,0),(245,179),(105,95,85),-1)
        cv2.circle(c,(160,40),37,(160,140,120),-1)
        return a,b,c

    def synthetic_analysis(self, frames, name, fps=24., manual=None):
        from unittest.mock import patch
        from overlay_core import analyze
        info=dict(kind='video',display_width=320,display_height=180,
                  duration=len(frames)/fps,fps=fps)
        with patch('overlay_core.video_frames',return_value=(f.copy() for f in frames)) as decoder:
            result, hit=analyze('synthetic', info, name, self.out(name), manual)
            self.assertFalse(hit)
            self.assertEqual(decoder.call_args.args[4], fps if manual is None else min(12.,fps))
        return result

    def test_39_dark_same_histogram_person_cut(self):
        from overlay_core import difference
        a,b,_=self.structural_scenes()
        a=(a.astype(float)*.75).astype(np.uint8)
        b=(b.astype(float)*.75).astype(np.uint8)
        d,h=difference(a,b)
        self.assertLess(d,.13); self.assertLess(h,.01)
        p=self.synthetic_analysis([a]*24+[b]*24,'dark_equal_hist')
        self.assertEqual([s['start'] for s in p['shots']],[0.,1.])
        self.assertEqual(p['cut_events'][0]['rule'],'spatial_edge_jump')

    def test_40_same_color_medium_to_closeup(self):
        a,_,c=self.structural_scenes()
        p=self.synthetic_analysis([a]*24+[c]*24,'same_color_closeup')
        self.assertEqual([s['start'] for s in p['shots']],[0.,1.])

    def test_41_outdoor_to_car_layout_cut(self):
        import cv2
        a=np.full((180,320,3),(30,70,45),np.uint8)
        cv2.rectangle(a,(10,10),(75,170),(105,95,85),-1)
        cv2.rectangle(a,(180,5),(220,179),(12,20,15),-1)
        b=np.full_like(a,(35,42,32))
        cv2.rectangle(b,(0,0),(319,65),(90,110,75),-1)
        cv2.rectangle(b,(80,70),(260,170),(105,95,85),-1)
        p=self.synthetic_analysis([a]*24+[b]*24,'outdoor_car')
        self.assertEqual([s['start'] for s in p['shots']],[0.,1.])

    def test_42_spatial_and_color_impulse_returns(self):
        a,b,c=self.structural_scenes()
        for impulse in (b,c,np.full_like(a,(174,135,138))):
            with self.subTest(mean=float(impulse.mean())):
                p=self.synthetic_analysis([a]*24+[impulse]*2+[a]*46,
                                          'return_'+str(int(impulse.mean())))
                self.assertEqual(len(p['shots']),1)
                self.assertGreater(p['detector_stats']['rejected_returns'],0)

    def test_43_flash_black_and_recovery_do_not_switch_dogs(self):
        a,b,_=self.structural_scenes()
        white=np.full_like(a,255); black=np.zeros_like(a)
        # Repeated opening pulses and an exposure recovery ramp.
        recovery=[np.clip(a.astype(float)*gain,0,255).astype(np.uint8) for gain in (3,2,1.5,1.2,1)]
        frames=[black]+[a]*4+[white]+[a]*5+[black]+[a]*12+[white]*2+recovery+[a]*24+[b]*24
        p=self.synthetic_analysis(frames,'flash_recovery')
        self.assertEqual([s['start'] for s in p['shots']],[0.,(len(frames)-24)/24])
        self.assertGreater(p['detector_stats']['illumination_frames'],0)

    def test_44_smooth_pan_is_not_a_cut(self):
        p=self.synthetic_analysis(self.textured_frames(72,2),'smooth_pan')
        self.assertEqual(len(p['shots']),1)

    def test_45_nearby_cuts_not_hidden_by_pending_confirmation(self):
        a,b,c=self.structural_scenes()
        for n in (6,8):
            with self.subTest(interval=n/24):
                p=self.synthetic_analysis([a]*24+[b]*n+[c]*24,'nearby_cuts_'+str(n))
                self.assertEqual([s['start'] for s in p['shots']],[0.,1.,(24+n)/24])

    def test_46_tail_precision_and_short_shot_policy(self):
        a,b,_=self.structural_scenes()
        p=self.synthetic_analysis([a]*24+[b]*6,'tail_quarter')
        self.assertEqual([s['start'] for s in p['shots']],[0.,1.])
        self.assertEqual(p['sampling_fps'],24.)
        self.assertAlmostEqual(p['cut_time_resolution_seconds'],1/24)
        self.assertIn('0.041667s',' '.join(p['limitations']))
        short=self.synthetic_analysis([a]*24+[b]*5,'tail_short')
        self.assertEqual(len(short['shots']),1)

    def test_47_encoded_same_color_cuts_plan_only(self):
        from overlay_core import PipeProcess
        from render_overlay import write_all
        a,b,c=self.structural_scenes()
        clip=self.base/'encoded_structural.mp4'
        enc=PipeProcess(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo',
                         '-pix_fmt','rgb24','-s','320x180','-r','24','-i','pipe:0',
                         '-c:v','libx264','-preset','ultrafast','-threads','1',clip],writer=True)
        try:
            for f in [a]*24+[b]*24+[c]*24: write_all(enc.p.stdin,f.tobytes())
            enc.finish()
        finally:
            enc.close()
        out=self.out('encoded_structural_plan')
        invoke(['--inputs',str(clip),'--output-dir',str(out),'--plan-only','--seed','42'])
        p=plans(out)[0]
        self.assertEqual(p['engine_version'],'overlay-1.3.5')
        self.assertEqual([s['start'] for s in p['shots']],[0.,1.,2.])
        self.assertTrue(all(a['dog_id']!=b['dog_id'] for a,b in zip(p['shots'],p['shots'][1:])))
        self.assertFalse(Path(p['output']).exists())


    def test_48_auto_image_larger_actual_canvas_and_inset(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement, PreparedDog
        blank=np.full((400,400,3),180,np.uint8)
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            large=placement(blank,dict(display_width=400,display_height=400),self.mock_dog(),None,'auto',random.Random(42))
            old=placement(blank,dict(display_width=400,display_height=400),self.mock_dog(),.24,'auto',random.Random(42))
        self.assertGreaterEqual(large['height'],96)
        self.assertLessEqual(large['height'],160)
        self.assertEqual(large['width'],large['height'])
        self.assertGreaterEqual(large['minimum_canvas_edge_margin'],.06)
        self.assertTrue(large['inset_preference_applied'])
        self.assertEqual(large['placement_mode'],'screen_sticker')
        self.assertFalse(large['edge_fallback'])
        big_alpha=PreparedDog(self.mock_dog(),large).frame(0)[1]
        old_alpha=PreparedDog(self.mock_dog(),old).frame(0)[1]
        self.assertGreater(np.count_nonzero(big_alpha),np.count_nonzero(old_alpha)*1.4)

    def test_49_auto_video_inset_respects_temporal_subjects(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement,intersects
        blank=np.full((400,400,3),180,np.uint8)
        obstacle=[0,0,250,400]
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',side_effect=[([],True),([obstacle],True)]):
            p=placement(blank,dict(display_width=400,display_height=400),self.mock_dog(),None,'auto',random.Random(4),extra_frames=[blank.copy()])
        self.assertGreaterEqual(p['height'],96)
        self.assertLessEqual(p['height'],160)
        self.assertGreaterEqual(p['minimum_canvas_edge_margin'],.06)
        self.assertEqual(p['protection_sample_count'],2)
        self.assertFalse(intersects([p['x'],p['y'],p['x']+p['width'],p['y']+p['height']],obstacle))

    def test_50_auto_rejects_space_below_size_floor(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement, intersects
        blank=np.full((400,400,3),180,np.uint8)
        obstacle=[80,0,400,400]
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',return_value=([obstacle],True)):
            with self.assertRaises(ValueError):
                placement(blank,dict(display_width=400,display_height=400),self.mock_dog(),None,'auto',random.Random(4))
            # Explicit small size remains an exception, without bypassing safety.
            p=placement(blank,dict(display_width=400,display_height=400),self.mock_dog(),.12,'auto',random.Random(4))
        self.assertLessEqual(p['width'],80)
        self.assertFalse(intersects([p['x'],p['y'],p['x']+p['width'],p['y']+p['height']],obstacle))

    def test_51_auto_keeps_large_edge_when_only_edge_is_safe(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement
        blank=np.full((400,400,3),180,np.uint8)
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',return_value=([[125,0,400,400]],True)):
            p=placement(blank,dict(display_width=400,display_height=400),self.mock_dog(),None,'auto',random.Random(4))
        self.assertGreaterEqual(p['height'],96)
        self.assertLessEqual(p['height'],160)
        self.assertLessEqual(p['x']+p['width'],125)
        self.assertTrue(p['edge_fallback'])
        self.assertFalse(p['size_fallback'])
        self.assertTrue(any('边缘fallback' in w for w in p['warnings']))

    def test_52_explicit_modes_preserve_size_anchor_and_constraints(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement,intersects
        blank=np.full((400,400,3),180,np.uint8)
        region=[[0,0,.5,1]]
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            fixed=placement(blank,dict(display_width=400,display_height=400),self.mock_dog(),.12,'auto',random.Random(4),[.75,.8],avoid_regions=region)
            corner=placement(blank,dict(display_width=400,display_height=400),self.mock_dog(),None,'corner',random.Random(4))
        self.assertEqual(fixed['height'],48)
        np.testing.assert_allclose(fixed['anchor_normalized'],[.75,.8],atol=1/400)
        self.assertFalse(fixed['inset_preference_applied'])
        self.assertFalse(corner['inset_preference_applied'])
        self.assertLessEqual(corner['height'],96)
        self.assertFalse(intersects([fixed['x'],fixed['y'],fixed['x']+fixed['width'],fixed['y']+fixed['height']],[0,0,200,400]))

    def test_53_cli_auto_image_video_report_new_policy(self):
        from PIL import Image
        photo=self.base/'blank_auto.png'
        Image.new('RGB',(640,400),(180,180,180)).save(photo)
        out=self.out('new_auto_policy')
        invoke(['--inputs',str(photo),str(self.fixtures/'three_shots.mp4'),'--output-dir',str(out),'--plan-only','--seed','42'])
        found={p['input']:p for p in plans(out) if 'media' in p}
        p=found[str(photo)]
        self.assertEqual(p['media']['duration'],5.)
        self.assertGreaterEqual(p['shots'][0]['height_ratio'],.24)
        self.assertLessEqual(p['shots'][0]['height_ratio'],.40)
        self.assertGreaterEqual(p['shots'][0]['placement']['minimum_canvas_edge_margin'],.06-1e-6)
        for p in found.values():
            self.assertEqual(p['engine_version'],'overlay-1.3.5')
            self.assertTrue(all(s['placement']['auto_size_range']==[.24,.40] for s in p['shots']))
            self.assertTrue(all(.24 <= s['height_ratio'] <= .40 for s in p['shots']))
            self.assertTrue(all(s['placement']['whole_canvas_bounds_checked'] for s in p['shots']))
            self.assertFalse(Path(p['output']).exists())


    def test_54_face_free_arm_and_neutral_skirt_protection(self):
        import cv2
        from classical_vision import body_cue_mask
        image=np.full((180,320,3),(15,45,25),np.uint8)
        cv2.rectangle(image,(70,80),(260,179),(145,145,145),-1)
        cv2.rectangle(image,(90,30),(125,105),(210,155,120),-1)
        cv2.rectangle(image,(120,55),(280,80),(210,155,120),-1)
        mask=body_cue_mask(image)
        self.assertGreater(mask[60:75,150:270].mean(),.95)
        self.assertGreater(mask[120:165,100:230].mean(),.95)
        self.assertLess(mask[:30,:40].mean(),.01)

    def test_55_extra_full_frame_body_union_is_hard_protection(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement,mask_box_fraction
        blank=np.full((400,400,3),180,np.uint8)
        mask=np.zeros((200,200),np.uint8); mask[90:,110:]=1
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            p=placement(blank,dict(display_width=400,display_height=400),self.mock_dog(),None,'auto',random.Random(4),extra_mask=mask)
        self.assertLessEqual(mask_box_fraction(mask,[p['x']/2,p['y']/2,(p['x']+p['width'])/2,(p['y']+p['height'])/2]),.015)


    def test_56_warm_fragmented_grass_does_not_bridge_into_skin(self):
        import cv2
        from classical_vision import body_cue_mask
        # Many warm blades pass the raw skin gate. Vary palette and exposure;
        # evidence must be local coherence, not a special grass RGB exclusion.
        yy,xx=np.indices((180,320))
        for warm in ((180,120,80),(140,95,65),(220,170,130)):
            with self.subTest(warm=warm):
                image=np.full((180,320,3),(60,90,40),np.uint8)
                image[(xx+2*yy)%5<3]=warm
                cv2.rectangle(image,(140,50),(158,145),(210,155,120),-1)
                cv2.rectangle(image,(150,65),(265,79),(210,155,120),-1)
                mask=body_cue_mask(image)
                self.assertLess(mask[:, :100].mean(),.01)
                self.assertGreater(mask[68:76,170:250].mean(),.95)
                self.assertLess(mask.mean(),.16)

    def test_57_neutral_growth_is_local_even_with_long_connected_ground(self):
        import cv2
        from classical_vision import body_cue_mask
        image=np.full((180,320,3),(20,65,35),np.uint8)
        # A neutral component below the global area gate touches skin but
        # extends much farther than this small seed's body scale.
        cv2.rectangle(image,(80,95),(300,150),(155,155,155),-1)
        cv2.rectangle(image,(60,75),(82,105),(210,155,120),-1)
        mask=body_cue_mask(image)
        self.assertGreater(mask[98:106,85:95].mean(),.95)
        self.assertLess(mask[110:145,180:290].mean(),.01)
        self.assertGreater(mask[82:100,65:78].mean(),.95)

    def test_58_skin_tones_and_textured_neutral_skirt_stay_continuous(self):
        import cv2
        from classical_vision import body_cue_mask
        for tone in ((65,42,32),(110,75,55),(170,120,90),(235,190,155)):
            for scale in (1,2):
                with self.subTest(tone=tone,scale=scale):
                    image=np.full((180,320,3),(15,45,25),np.uint8)
                    yy,xx=np.indices((100,191))
                    # Low-contrast garment folds, not one exact grey value.
                    folds=(145+12*np.sin(xx/9)+8*np.cos(yy/11)).astype(np.uint8)
                    image[80:180,70:261]=folds[...,None]
                    cv2.rectangle(image,(90,30),(125,105),tone,-1)
                    cv2.rectangle(image,(120,55),(280,80),tone,-1)
                    image=cv2.resize(image,(320*scale,180*scale))
                    mask=cv2.resize(body_cue_mask(image),(320,180),interpolation=cv2.INTER_NEAREST)
                    self.assertGreater(mask[60:75,150:270].mean(),.95)
                    self.assertGreater(mask[120:165,100:230].mean(),.95)
                    self.assertLess(mask[:30,:40].mean(),.01)

    def test_59_distant_skin_padding_does_not_consume_inset_canvas(self):
        import random
        import cv2
        from unittest.mock import patch
        from classical_vision import body_cue_mask
        from overlay_core import placement,mask_box_fraction
        image=np.full((640,480,3),(30,70,35),np.uint8)
        cv2.rectangle(image,(320,200),(331,249),(210,155,120),-1)
        mask=body_cue_mask(image)
        # A small observed arm retains a contour halo, not 16px frame padding.
        self.assertGreater(mask[210:240,322:329].mean(),.95)
        self.assertEqual(int(mask[220,307]),0)
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            p=placement(image,dict(display_width=480,display_height=640),self.mock_dog(),None,'auto',random.Random(42))
        self.assertGreaterEqual(p['height']/640,.24)
        self.assertGreaterEqual(p['minimum_canvas_edge_margin'],.06-1e-6)
        self.assertLessEqual(mask_box_fraction(mask,[p['x'],p['y'],p['x']+p['width'],p['y']+p['height']]),.015)


    def test_60_warm_wall_and_high_texture_fur_do_not_seal_mask(self):
        from classical_vision import body_cue_mask
        wall=np.full((180,320,3),(170,110,70),np.uint8)
        self.assertLess(body_cue_mask(wall).mean(),.01)
        yy,xx=np.indices((180,320))
        fur=np.full_like(wall,(190,130,90))
        fur[(xx+yy)%2==0]=(110,65,40)
        self.assertLess(body_cue_mask(fur).mean(),.05)

    def test_61_failed_cli_retains_shot_and_layered_diagnostics(self):
        out=self.out('failure_diagnostics')
        r=invoke(['--inputs',str(self.fixtures/'three_shots.mp4'),'--output-dir',str(out),
                  '--plan-only','--seed','42','--avoid-region','0','0','1','1'],expect_ok=False)
        self.assertNotEqual(r.returncode,0)
        p=plans(out)[0]
        self.assertEqual(p['status'],'failed')
        self.assertEqual(p['first_failed_shot']['start'],0.)
        self.assertGreater(p['first_failed_shot']['end'],0.)
        d=p['first_failed_shot']['protection_diagnostics']
        d=d.get('window',d)
        self.assertTrue(d['evaluated_sizes'])
        self.assertTrue(any(s['rejection_counts'].get('user_region',0)>0 for s in d['evaluated_sizes']))
        self.assertFalse(Path(p['output']).exists())

    def test_62_temporal_placement_keeps_same_dog_and_exact_boundary(self):
        from render_overlay import active_placement
        from motion_tracking import plan_tracking
        a=dict(placement_mode='screen_sticker',x=0,y=0,width=10,height=10)
        b=dict(placement_mode='screen_sticker',x=90,y=0,width=10,height=10)
        shot=dict(start=0.,end=1.,frame_start=0,dog_id='one',placement=a,
                  placement_segments=[dict(frame_start=0,placement=a),dict(frame_start=15,placement=b)])
        self.assertIs(active_placement(shot,14),a)
        self.assertIs(active_placement(shot,15),b)
        p=dict(media=dict(kind='video',fps=30,display_width=100,display_height=100),tracking='flow',requested_anchor=None,shots=[shot])
        plan_tracking('not-decoded',p,30)
        self.assertEqual(shot['motion']['mode'],'off')
        self.assertEqual(shot['motion']['reason'],'temporal_safe_windows')
        self.assertEqual(shot['dog_id'],'one')

    def test_63_repeated_mirror_impulses_do_not_become_dense_cuts(self):
        a,b,_=self.structural_scenes()
        frames=[a]*24
        for _ in range(8): frames += [b]*2+[a]*10
        p=self.synthetic_analysis(frames,'repeated_mirror_pulses')
        self.assertEqual(len(p['shots']),1)


    def test_64_special_fallback_requires_explicit_opt_in(self):
        from render_overlay import parser
        self.assertFalse(parser().parse_args([]).allow_special_fallback)
        self.assertTrue(parser().parse_args(['--allow-special-fallback']).allow_special_fallback)

    def test_65_relaxed_policy_preserves_manual_regions_and_bounds(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement
        image=np.full((180,320,3),80,np.uint8)
        info=dict(display_width=320,display_height=180)
        with patch('overlay_core.face_boxes',return_value=([[0,0,320,180]],True)), patch('overlay_core.person_boxes',return_value=([[0,0,320,180]],True)):
            with self.assertRaises(ValueError):
                placement(image,info,self.mock_dog(),.15,'auto',random.Random(42),protection_policy='body-relaxed')
            p=placement(image,info,self.mock_dog(),.15,'auto',random.Random(42),protection_policy='detected-relaxed')
            self.assertEqual(p['protection_policy'],'detected-relaxed')
            self.assertGreaterEqual(p['x'],0);self.assertGreaterEqual(p['y'],0)
            self.assertLessEqual(p['x']+p['width'],320);self.assertLessEqual(p['y']+p['height'],180)
            self.assertEqual(p['placement_mode'],'screen_sticker')
            with self.assertRaises(ValueError):
                placement(image,info,self.mock_dog(),.15,'auto',random.Random(42),avoid_regions=[[0,0,1,1]],protection_policy='detected-relaxed')

    def test_66_large_authorized_partial_overlap_is_recorded(self):
        import random
        from render_overlay import special_fallback,parser
        from unittest.mock import patch
        image=np.full((180,320,3),80,np.uint8)
        args=parser().parse_args(['--allow-special-fallback'])
        with patch('overlay_core.face_boxes',return_value=([[0,0,320,180]],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            p=special_fallback(image,dict(display_width=320,display_height=180),self.mock_dog(),random.Random(42),args,[])
        self.assertTrue(p['special_scene_fallback'])
        self.assertTrue(p['partial_overlap_accepted'])
        self.assertNotIn('small_size_accepted',p)
        self.assertGreaterEqual(p['actual_height_ratio'],.24)
        self.assertLessEqual(p['actual_height_ratio'],.40)
        self.assertEqual(p['auto_size_range'],[.24,.40])
        self.assertEqual(p['protection_policy'],'detected-relaxed')
        self.assertGreaterEqual(p['minimum_canvas_edge_margin'],.06)
        self.assertTrue(p['inset_preference_applied'])
        self.assertFalse(p['edge_fallback'])
        self.assertTrue(p['whole_canvas_bounds_checked'])

    def test_67_wide_action_auto_cannot_fit_below_floor(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement
        image=np.full((400,100,3),180,np.uint8)
        dog=self.mock_dog()
        dog.update(width=200,height=100,anchor=(100,96),union_bbox=[3,3,197,96])
        dog['frames']=np.zeros((3,100,200,4),np.uint8)
        dog['frames'][:,3:96,3:197]=[210,80,30,255]
        info=dict(display_width=100,display_height=400)
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            for policy in ('strict','body-relaxed','detected-relaxed'):
                with self.subTest(policy=policy), self.assertRaises(ValueError):
                    placement(image,info,dog,None,'auto',random.Random(42),protection_policy=policy)
            explicit=placement(image,info,dog,.12,'auto',random.Random(42))
        self.assertLess(explicit['actual_height_ratio'],.24)
        self.assertLessEqual(explicit['x']+explicit['width'],100)
        self.assertTrue(explicit['whole_canvas_bounds_checked'])

    def test_68_special_fallback_never_bypasses_manual_regions(self):
        import random
        from unittest.mock import patch
        from overlay_core import intersects
        from render_overlay import special_fallback,parser
        image=np.full((400,400,3),180,np.uint8)
        info=dict(display_width=400,display_height=400)
        args=parser().parse_args(['--allow-special-fallback','--avoid-region','0','0','.5','1'])
        with patch('overlay_core.face_boxes',return_value=([[0,0,400,400]],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            p=special_fallback(image,info,self.mock_dog(),random.Random(42),args,[])
            args.avoid_region=[[0,0,1,1]]
            with self.assertRaises(ValueError):
                special_fallback(image,info,self.mock_dog(),random.Random(42),args,[])
        rect=[p['x'],p['y'],p['x']+p['width'],p['y']+p['height']]
        self.assertFalse(intersects(rect,[0,0,200,400]))
        self.assertGreaterEqual(p['actual_height_ratio'],.24)
        self.assertLessEqual(p['actual_height_ratio'],.40)
        self.assertTrue(p['whole_canvas_bounds_checked'])

    def test_69_body_relaxed_precedes_all_detected_relaxed(self):
        import random
        from unittest.mock import patch
        from overlay_core import intersects
        from render_overlay import special_fallback,parser
        image=np.full((400,400,3),180,np.uint8)
        info=dict(display_width=400,display_height=400)
        face=[0,0,60,60]; text=[340,0,400,60]
        args=parser().parse_args(['--allow-special-fallback'])
        with patch('overlay_core.face_boxes',return_value=([face],True)), patch('overlay_core.person_boxes',return_value=([[0,0,400,400]],True)), patch('overlay_core.text_regions',return_value=[text]):
            p=special_fallback(image,info,self.mock_dog(),random.Random(42),args,[])
        self.assertEqual(p['protection_policy'],'body-relaxed')
        rect=[p['x'],p['y'],p['x']+p['width'],p['y']+p['height']]
        self.assertFalse(any(intersects(rect,b) for b in (face,text)))
        self.assertTrue(p['partial_overlap_accepted'])
        self.assertGreaterEqual(p['actual_height_ratio'],.24)
        self.assertLessEqual(p['actual_height_ratio'],.40)
        self.assertGreaterEqual(p['minimum_canvas_edge_margin'],.06)
        self.assertEqual(p['placement_mode'],'screen_sticker')


    def test_70_edge_body_candidate_retries_detected_relaxed_inset(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement
        from render_overlay import special_fallback,parser
        image=np.full((400,400,3),180,np.uint8)
        info=dict(display_width=400,display_height=400)
        face=[125,0,400,400]
        args=parser().parse_args(['--allow-special-fallback'])
        with patch('overlay_core.face_boxes',return_value=([face],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            body=placement(image,info,self.mock_dog(),None,'auto',random.Random(42),protection_policy='body-relaxed')
            p=special_fallback(image,info,self.mock_dog(),random.Random(42),args,[])
        self.assertLess(body['minimum_canvas_edge_margin'],.055)
        self.assertEqual(p['protection_policy'],'detected-relaxed')
        self.assertGreaterEqual(p['minimum_canvas_edge_margin'],.055)
        self.assertGreater(p['minimum_canvas_edge_margin'],body['minimum_canvas_edge_margin'])
        self.assertGreaterEqual(p['actual_height_ratio'],.24)
        self.assertLessEqual(p['actual_height_ratio'],.40)
        self.assertTrue(p['partial_overlap_accepted'])
        self.assertTrue(p['whole_canvas_bounds_checked'])

    def test_71_special_inset_threshold_and_explicit_exceptions(self):
        import random
        from unittest.mock import patch
        from render_overlay import special_fallback,parser
        image=np.full((400,400,3),180,np.uint8)
        info=dict(display_width=400,display_height=400)
        # Isolate policy routing at the threshold from CV/geometry heuristics.
        for margin, second, flags, expected, count in (
                (.055,.10,[],'body-relaxed',1),
                (.054,.10,[],'detected-relaxed',2),
                (0.,.03,[],'detected-relaxed',2),
                (.03,0.,[],'body-relaxed',2),
                (0.,.10,['--height-ratio','.12'],'body-relaxed',1),
                (0.,.10,['--anchor','.5','.8'],'body-relaxed',1),
                (0.,.10,['--placement-mode','corner'],'body-relaxed',1)):
            with self.subTest(margin=margin,second=second,flags=flags):
                args=parser().parse_args(['--allow-special-fallback',*flags])
                candidates=[dict(minimum_canvas_edge_margin=m,joint_cost=0.,
                                 evaluated_height_ratio=.28,protection_policy=policy)
                            for m,policy in ((margin,'body-relaxed'),(second,'detected-relaxed'))]
                with patch('render_overlay.placement',side_effect=candidates) as search:
                    p=special_fallback(image,info,self.mock_dog(),random.Random(42),args,[])
                self.assertEqual(p['protection_policy'],expected)
                self.assertEqual(search.call_count,count)
                self.assertEqual([c.kwargs['protection_policy'] for c in search.call_args_list],
                                 ['body-relaxed','detected-relaxed'][:count])
                for call in search.call_args_list:
                    self.assertEqual(call.args[3],args.height_ratio)
                    self.assertEqual(call.args[4],args.placement_mode)
                    self.assertEqual(call.args[6],args.anchor)


class LosslessCacheTests(unittest.TestCase):
    """Targeted codec/maintenance tests; no media fixture generation/rendering."""
    def setUp(self):
        import zipfile
        from lossless_cache import sha256
        work = ROOT/'temp_data/lossless'
        work.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='unit-', dir=work)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.assets = self.base/'assets'
        (self.assets/'dogs').mkdir(parents=True)
        self.work = self.base/'work'
        self.frames = np.zeros((12,96,96,4), np.uint8)
        patch = np.random.default_rng(41).integers(0,256,(72,72,4),np.uint8)
        patch[...,3] = 255
        self.frames[:,12:84,12:84] = patch
        self.path = self.assets/'dogs/dog_01.npz'
        np.savez_compressed(self.path, frames=self.frames)
        self.manifest_path = self.assets/'dog_manifest.json'
        self.manifest = {'version':1,'dogs':[dict(
            id='dog_01',file='dogs/dog_01.npz',frame_count=12,width=96,height=96,
            fps=30.,anchor=[48.,84.],dominant_color=[111,122,133],
            sha256=sha256(self.path),npz_bytes=self.path.stat().st_size,
            source='source/dog_01.mp4',source_sha256='source-never-modified',
            experimental_metadata={'keep':[1,2,3]})]}
        self.save_manifest()

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest),encoding='utf-8')

    def test_01_repack_all_members_arrays_and_hidden_rgba(self):
        import zipfile
        from lossless_cache import compare_npz, repack_npz
        # Include non-array bytes and a second safe NPY (including NaN).
        import io
        buffer = io.BytesIO()
        np.save(buffer,np.array([1.,np.nan]),allow_pickle=False)
        with zipfile.ZipFile(self.path,'a') as z:
            z.writestr('note.txt',b'extra member content must survive\x00')
            z.writestr('other.npy',buffer.getvalue())
        target = self.base/'new.npz'
        repack_npz(self.path,target)
        result = compare_npz(self.path,target)
        self.assertTrue(result['all_members_byte_identical'])
        self.assertTrue(result['all_rgba_pixels_equal'])
        self.assertEqual(len(result['old_members']),3)
        self.assertEqual(len(result['arrays']),2)
        self.assertEqual(len(result['frame_pixels_sha256']),12)

    def test_02_manifest_preservation_and_idempotency(self):
        from lossless_cache import compress_assets, cache_metadata_keys
        before = json.loads(self.manifest_path.read_text())
        first = compress_assets(self.assets,self.work)
        self.assertGreater(first['saved_bytes'],0)
        after = json.loads(self.manifest_path.read_text())
        excluded = {'sha256','npz_bytes',*cache_metadata_keys()}
        self.assertEqual({k:v for k,v in before['dogs'][0].items() if k not in excluded},
                         {k:v for k,v in after['dogs'][0].items() if k not in excluded})
        snapshot = [(p.read_bytes(),p.stat().st_mtime_ns) for p in (self.path,self.manifest_path)]
        second = compress_assets(self.assets,self.work,self.work/'rerun.json')
        self.assertTrue(second['dogs'][0]['skipped_already_lzma'])
        self.assertEqual(second['saved_bytes'],0)
        self.assertEqual(snapshot,[(p.read_bytes(),p.stat().st_mtime_ns) for p in (self.path,self.manifest_path)])

    def test_03_existing_hash_mismatch_refuses(self):
        from lossless_cache import compress_assets
        self.manifest['dogs'][0]['sha256']='0'*64
        self.save_manifest()
        before = self.path.read_bytes(),self.manifest_path.read_bytes()
        with self.assertRaisesRegex(ValueError,'hash/size mismatch'):
            compress_assets(self.assets,self.work)
        self.assertEqual(before,(self.path.read_bytes(),self.manifest_path.read_bytes()))

    def test_04_corrupt_npz_even_matching_hash_refuses(self):
        from lossless_cache import compress_assets,sha256
        self.path.write_bytes(b'not a ZIP file')
        self.manifest['dogs'][0].update(sha256=sha256(self.path),npz_bytes=self.path.stat().st_size)
        self.save_manifest()
        before=self.path.read_bytes(),self.manifest_path.read_bytes()
        import zipfile
        with self.assertRaises(zipfile.BadZipFile):
            compress_assets(self.assets,self.work)
        self.assertEqual(before,(self.path.read_bytes(),self.manifest_path.read_bytes()))

    def test_05_failed_comparison_never_replaces(self):
        from unittest.mock import patch
        from lossless_cache import compress_assets
        before=self.path.read_bytes(),self.manifest_path.read_bytes()
        with patch('lossless_cache.compare_npz',side_effect=ValueError('injected validation failure')):
            with self.assertRaisesRegex(ValueError,'validation failure'):
                compress_assets(self.assets,self.work)
        self.assertEqual(before,(self.path.read_bytes(),self.manifest_path.read_bytes()))

    def test_06_manifest_install_failure_rolls_back_cache(self):
        from unittest.mock import patch
        from lossless_cache import compress_assets,atomic_json
        before=self.path.read_bytes(),self.manifest_path.read_bytes()
        def fail_manifest(path,value,work):
            if path == self.manifest_path:
                raise OSError('injected manifest install failure')
            return atomic_json(path,value,work)
        with patch('lossless_cache.atomic_json',side_effect=fail_manifest):
            with self.assertRaisesRegex(OSError,'install failure'):
                compress_assets(self.assets,self.work)
        self.assertEqual(before,(self.path.read_bytes(),self.manifest_path.read_bytes()))

    def test_07_pickle_and_duplicate_members_refused(self):
        import zipfile,warnings
        from lossless_cache import inspect_npz
        bad=self.base/'pickle.npz'
        np.savez_compressed(bad,frames=self.frames,unsafe=np.array([{}],dtype=object))
        with self.assertRaises(ValueError): inspect_npz(bad)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(self.path,'a') as z: z.writestr('frames.npy',b'bad duplicate')
        with self.assertRaisesRegex(ValueError,'duplicate'): inspect_npz(self.path)

    def test_08_new_preparation_writer_is_lzma_and_pickle_free(self):
        import zipfile
        from prepare_assets import write_npz_lzma
        target=self.base/'fresh.npz'
        write_npz_lzma(target,frames=self.frames)
        with zipfile.ZipFile(target) as z:
            self.assertEqual([i.compress_type for i in z.infolist()],[14])
        with np.load(target,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['frames'],self.frames)
        with self.assertRaises(ValueError):
            write_npz_lzma(self.base/'unsafe.npz',frames=np.array([{}],dtype=object))

    def test_09_existing_decoded_hash_mismatch_refuses(self):
        from lossless_cache import compress_assets
        self.manifest['dogs'][0]['decoded_frames_sha256']='0'*64
        self.save_manifest()
        before=self.path.read_bytes()
        with self.assertRaisesRegex(ValueError,'Decoded frames hash mismatch'):
            compress_assets(self.assets,self.work)
        self.assertEqual(before,self.path.read_bytes())

    def test_10_all_16_lzma_caches_load_via_unchanged_engine(self):
        import zipfile
        from overlay_core import DogLibrary
        from lossless_cache import sha256,frames_hash,COMPRESSION
        library=DogLibrary(ROOT/'assets')
        self.assertEqual(len(library.entries),16)
        total=0
        for dog_id,entry in library.entries.items():
            path=ROOT/'assets'/entry['file']
            with zipfile.ZipFile(path) as z:
                self.assertTrue(all(i.compress_type==14 for i in z.infolist()))
            self.assertEqual(sha256(path),entry['sha256'])
            self.assertEqual(path.stat().st_size,entry['npz_bytes'])
            self.assertEqual(entry['compression'],COMPRESSION)
            dog=library.load(dog_id)
            self.assertEqual(dog['frames'].shape,(entry['frame_count'],entry['height'],entry['width'],4))
            self.assertEqual(frames_hash(dog['frames']),entry['decoded_frames_sha256'])
            total+=entry['frame_count']
        self.assertEqual(total,827)


    def test_11_real_pixel_change_is_rejected(self):
        from lossless_cache import write_npz_lzma,compare_npz
        changed=self.frames.copy()
        changed[0,0,0,0]=1  # Even RGB beneath zero alpha is protected.
        target=self.base/'changed.npz'
        write_npz_lzma(target,frames=changed)
        with self.assertRaisesRegex(ValueError,'member content'):
            compare_npz(self.path,target)

    def test_12_entire_batch_preflight_before_first_replacement(self):
        import copy
        from lossless_cache import compress_assets,sha256
        bad=self.assets/'dogs/dog_02.npz'
        bad.write_bytes(b'broken')
        second=copy.deepcopy(self.manifest['dogs'][0])
        second.update(id='dog_02',file='dogs/dog_02.npz',sha256=sha256(bad),npz_bytes=bad.stat().st_size)
        self.manifest['dogs'].append(second)
        self.save_manifest()
        before=self.path.read_bytes(),bad.read_bytes(),self.manifest_path.read_bytes()
        import zipfile
        with self.assertRaises(zipfile.BadZipFile): compress_assets(self.assets,self.work)
        self.assertEqual(before,(self.path.read_bytes(),bad.read_bytes(),self.manifest_path.read_bytes()))

    def test_13_concurrent_maintenance_lock_refuses(self):
        from lossless_cache import compress_assets
        self.work.mkdir()
        lock=self.work/'maintenance.lock'
        lock.write_bytes(b'other maintenance')
        with self.assertRaises(FileExistsError): compress_assets(self.assets,self.work)
        self.assertEqual(lock.read_bytes(),b'other maintenance')


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--report',default=str(ROOT/'temp_data/test_results.json'))
    a.add_argument('--real-summary',help='optional seed42/hd eight-real-input batch summary for functional regression')
    a.add_argument('--lossless-only',action='store_true',help='only NPZ codec/maintenance tests; never generate media or render')
    args=a.parse_args()
    OverlayTests.real_summary=args.real_summary
    start=time.perf_counter()
    suite=unittest.TestSuite()
    if not args.lossless_only:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(OverlayTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(LosslessCacheTests))
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'ok':r.wasSuccessful(),'tests_run':r.testsRun,'failures':len(r.failures),
            'errors':len(r.errors),'elapsed_seconds':round(time.perf_counter()-start,3),
            'details':[{'test':str(t),'trace':msg} for t,msg in r.failures+r.errors]}
    path=Path(args.report); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))
    sys.exit(0 if r.wasSuccessful() else 1)


if __name__=='__main__':
    main()
