import numpy as np
from quacknet import Augementation
import tempfile
from PIL import Image
import os

def test_flip():
    aug = Augementation()

    img = np.array([[[1, 0, 0]] * 2] * 2)
    images = np.array([img])
    labels = np.array([[[1, 0, 0]] * 2] * 2)

    augmetedImages, augmetedLabels = aug.dataAugmentation(images, labels)

    assert augmetedImages.shape == (4, 2, 2, 3)
    assert np.allclose(augmetedImages[0], img)
    assert np.allclose(augmetedImages[1], np.fliplr(img))
    assert np.allclose(augmetedImages[2], np.flipud(img))
    assert np.allclose(augmetedImages[3], np.flipud(np.fliplr(img)))

    assert augmetedLabels.shape == (4, 2, 2, 3)
    assert np.allclose(augmetedLabels[0], img)
    assert np.allclose(augmetedLabels[1], np.fliplr(img))
    assert np.allclose(augmetedLabels[2], np.flipud(img))
    assert np.allclose(augmetedLabels[3], np.flipud(np.fliplr(img)))

def test_hotEncodeLabels():
    labels = [0, 2, 1]
    numClasses = 3
    expected = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])
    result = Augementation.hotEncodeLabels(None, labels, numClasses)
    assert np.allclose(result, expected)

def test_preprocessImages():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "img.jpg")
        Image.new('RGB', (50, 50), color=(100, 150, 200)).save(path)
        result = Augementation.preprocessImages(None, [path], targetSize=(32, 32))
        assert result.shape == (1, 32, 32, 3)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

def test_getImagePaths():
    with tempfile.TemporaryDirectory() as tmpdir:
        img1 = os.path.join(tmpdir, "a.jpg")
        img2 = os.path.join(tmpdir, "b.png")
        Image.new('RGB', (10, 10)).save(img1)
        Image.new('RGB', (10, 10)).save(img2)
        result = Augementation.getImagePaths(None, tmpdir)
        assert img1 in result
        assert img2 in result
        assert len(result) == 2