import base64

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src import preprocess as P
from src.dataset import CLASSES, ISLVideoDataset, choose_val_subjects, parse_subject_take, split_by_subject
from src.engine import DEFAULT_HPARAMS, make_model
from src.inference import FrameWindow, load_model, predict_clip, sample_clip
from src.train import save_checkpoint


def fake_sample(class_idx, subject, take, n=20, seed=0, as_numpy=False):
    frames = np.random.default_rng(seed).integers(0, 256, size=(n, P.IMG_SIZE, P.IMG_SIZE, 3), dtype=np.uint8)
    return {
        'frames': frames if as_numpy else torch.from_numpy(frames),
        'total_frames': n,
        'fps': 30.0,
        'label': class_idx,
        'class_name': CLASSES[class_idx],
        'subject': subject,
        'video_name': f'{CLASSES[class_idx]}_Crop_{subject}_{take}.avi',
    }


# ---------- splitting ----------

def test_parse_subject_take():
    assert parse_subject_take('help_Crop_001_02.avi') == ('001', '02')
    assert parse_subject_take('data/raw/pain/pain_crop_026_01.mp4') == ('026', '01')
    with pytest.raises(ValueError):
        parse_subject_take('help.avi')


def test_subject_split_keeps_each_signer_in_one_split():
    samples = [fake_sample(c, f'{s:03d}', f'{t:02d}', n=4)
               for c in range(5) for s in range(1, 11) for t in (1, 2)]
    test_subjects = ['009', '010']
    remaining = [f'{s:03d}' for s in range(1, 9)]
    val_subjects = choose_val_subjects(remaining, 2, seed=0)
    train, val, test = split_by_subject(samples, test_subjects, val_subjects)

    subj = lambda xs: {s['subject'] for s in xs}
    assert subj(test) == set(test_subjects)
    assert subj(val) == set(val_subjects)
    assert not (subj(train) & subj(val)) and not (subj(train) & subj(test)) and not (subj(val) & subj(test))
    assert len(train) + len(val) + len(test) == len(samples)
    assert len(test) == 2 * 5 * 2  # both takes of both test signers, all classes


def test_split_rejects_unknown_or_overlapping_subjects():
    samples = [fake_sample(0, '001', '01', n=4), fake_sample(0, '002', '01', n=4)]
    with pytest.raises(ValueError):
        split_by_subject(samples, ['099'], [])
    with pytest.raises(ValueError):
        split_by_subject(samples, ['001'], ['001'])


# ---------- preprocessing parity ----------

def test_dataset_eval_path_matches_live_path():
    """Training/eval and the webcam/API must turn the same video into the same tensor."""
    rng = np.random.default_rng(1)
    video_bgr = rng.integers(0, 256, size=(50, 240, 320, 3), dtype=np.uint8)
    prepared = np.stack([P.prepare_frame(f) for f in video_bgr])
    sample = {'frames': torch.from_numpy(prepared), 'total_frames': 50, 'label': 0,
              'video_name': 'help_Crop_001_01.avi'}

    dataset_tensor, _, _ = ISLVideoDataset([sample], is_train=False)[0]
    live_tensor = P.to_model_tensor(sample_clip(list(prepared), P.NUM_FRAMES))
    assert dataset_tensor.shape == (P.NUM_FRAMES, 3, P.IMG_SIZE, P.IMG_SIZE)
    assert torch.allclose(dataset_tensor, live_tensor)


def test_dataset_accepts_old_numpy_cache_and_augments():
    sample = fake_sample(2, '003', '01', n=30, as_numpy=True)
    x, label, _ = ISLVideoDataset([sample], is_train=True)[0]
    assert x.shape == (P.NUM_FRAMES, 3, P.IMG_SIZE, P.IMG_SIZE) and label == 2


@pytest.mark.parametrize('w,h', [(640, 480), (480, 640), (1920, 1080), (112, 112)])
def test_roi_box_stays_inside_frame(w, h):
    x0, y0, side = P.roi_box(w, h)
    assert 0 <= x0 and 0 <= y0 and x0 + side <= w and y0 + side <= h
    assert side == round(P.DEFAULT_ROI['size'] * min(w, h))


def test_frame_window_covers_a_whole_gesture():
    window = FrameWindow(seconds=2.0)
    for i in range(30):              # 1 s of frames: not enough yet
        window.add(i, i / 30)
    assert not window.ready(16)
    for i in range(30, 120):         # 4 s total
        window.add(i, i / 30)
    assert window.ready(16)
    assert window.span() <= 2.0
    sampled = sample_clip(window.frames(), 16)
    assert sampled[-1] == 119 and sampled[0] >= 119 - 60   # spread over the last 2 s, not the last 16 frames
    assert sampled[-1] - sampled[0] > 50


# ---------- model ----------

def test_model_forward_shape():
    model = make_model(len(CLASSES), DEFAULT_HPARAMS).eval()
    with torch.no_grad():
        logits, attn = model(torch.randn(2, P.NUM_FRAMES, 3, P.IMG_SIZE, P.IMG_SIZE))
    assert logits.shape == (2, len(CLASSES))
    assert attn.shape == (2, P.NUM_FRAMES, 1)


@pytest.fixture
def model_path(tmp_path):
    torch.manual_seed(0)
    model = make_model(len(CLASSES), DEFAULT_HPARAMS)
    path = tmp_path / 'model.pth'
    split = {'train_subjects': ['001'], 'val_subjects': ['002'], 'test_subjects': ['003']}
    save_checkpoint(str(path), model.state_dict(), CLASSES, DEFAULT_HPARAMS,
                    {'epoch': 1, 'val_acc': 50.0, 'val_loss': 1.0}, split, 2.4, seed=0)
    return path


def test_checkpoint_loads_safely(model_path):
    lm = load_model(model_path, 'cpu')   # weights_only=True inside
    assert lm.classes == CLASSES and lm.window_seconds == pytest.approx(2.4)
    assert lm.meta['split']['test_subjects'] == ['003']
    probs, attn = predict_clip(lm, np.zeros((P.NUM_FRAMES, P.IMG_SIZE, P.IMG_SIZE, 3), np.uint8))
    assert probs.shape == (len(CLASSES),) and attn.shape == (P.NUM_FRAMES,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


# ---------- API ----------

@pytest.fixture
def api(monkeypatch):
    import app.main as main
    main._state.update(lm=None, error=None)
    yield main
    main._state.update(lm=None, error=None)


@pytest.fixture
def client(api, model_path, monkeypatch):
    monkeypatch.setattr(api, 'MODEL_PATH', model_path)
    with TestClient(api.app) as c:
        yield c


def jpeg_b64(img):
    ok, buf = cv2.imencode('.jpg', img)
    assert ok
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


def test_api_info(client):
    info = client.get('/api/info').json()
    assert info['model_loaded'] and info['supported_classes'] == CLASSES
    assert info['window_seconds'] == pytest.approx(2.4)
    assert info['roi'] == P.DEFAULT_ROI


def test_predict_frames_idle(client):
    still = np.full((240, 320, 3), 128, np.uint8)
    r = client.post('/api/predict-frames', json={'frames': [jpeg_b64(still)] * 16})
    assert r.status_code == 200 and r.json()['is_idle'] is True


def test_predict_frames_moving(client):
    rng = np.random.default_rng(0)
    frames = [jpeg_b64(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)) for _ in range(16)]
    r = client.post('/api/predict-frames', json={'frames': frames})
    body = r.json()
    assert r.status_code == 200 and body['is_idle'] is False
    assert body['prediction'] in CLASSES + ['detecting', 'none']
    assert set(body['class_probabilities']) == set(CLASSES)


def test_predict_frames_rejects_bad_input(client):
    assert client.post('/api/predict-frames', json={'frames': ['not base64!!'] * 16}).status_code == 400
    assert client.post('/api/predict-frames', json={'frames': ['AAAA'] * 65}).status_code == 422
    assert client.post('/api/predict-frames', json={'frames': []}).status_code == 422


def test_predict_upload(client, tmp_path):
    path = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (160, 120))
    rng = np.random.default_rng(0)
    for _ in range(20):
        writer.write(rng.integers(0, 256, (120, 160, 3), dtype=np.uint8))
    writer.release()

    with open(path, 'rb') as f:
        r = client.post('/api/predict', files={'file': ('clip.avi', f, 'video/x-msvideo')})
    body = r.json()
    assert r.status_code == 200, body
    assert body['prediction'] in CLASSES and body['total_frames_in_video'] == 20
    assert len(body['attention_weights']) == P.NUM_FRAMES


def test_unreadable_upload_is_rejected(client):
    r = client.post('/api/predict', files={'file': ('x.avi', b'not a video', 'video/x-msvideo')})
    assert r.status_code == 400


def test_missing_model_returns_503(api, tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'MODEL_PATH', tmp_path / 'missing.pth')
    with TestClient(api.app) as c:
        assert c.get('/api/info').json()['model_loaded'] is False
        r = c.post('/api/predict-frames', json={'frames': ['AAAA'] * 16})
        assert r.status_code == 503
