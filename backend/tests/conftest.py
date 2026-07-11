from unittest.mock import patch

# Keep app.main imports fast during API tests without masking get_model().
_torch_load_patch = patch("torch.load", return_value={})
_load_state_dict_patch = patch("torch.nn.Module.load_state_dict", return_value=None)
_torch_load_patch.start()
_load_state_dict_patch.start()
