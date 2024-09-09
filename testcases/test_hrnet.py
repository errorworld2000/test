from models.hrnet import hrnet18

class TestHRNet:
    def test_hrnet18(self):
        model=hrnet18(pretrained=False)
        assert model is not None