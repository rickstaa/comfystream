import torch

print("Loading pre-startup script for controlnet torch.compile() compatibility...")

def patch_controlnet_for_stream():
    """Patch ControlNet for better compatibility with torch.compile()"""
    try:
        from comfy.controlnet import ControlBase
        original_control_merge = ControlBase.control_merge

        def safe_clone(t):
            return t.clone() if (t is not None and t.requires_grad) else t

        def wrapped_control_merge(self, control, control_prev, output_dtype):
            # Mark CUDA graph step at start and ensure synchronization
            if torch.cuda.is_available():
                mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
                if mark_step_begin:
                    mark_step_begin()
                torch.cuda.synchronize()

            # Deep clone control outputs to prevent CUDA graph overwrites
            control = {k: [safe_clone(t) for t in v] for k, v in control.items()}
           
            # Deep clone previous control if it exists
            if control_prev is not None:
                control_prev = {k: [safe_clone(t) for t in v] for k, v in control_prev.items()}

            try:
                # Get result from original merge function
                result = original_control_merge(self, control, control_prev, output_dtype)
            
                # Clone all output tensors to prevent overwriting
                result = {k: [safe_clone(t) for t in v] for k, v in result.items()}
            finally:
                # Ensure synchronization at the end
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            return result
           
        # Apply the patch
        ControlBase.control_merge = wrapped_control_merge
        print("Successfully patched ControlNet for torch.compile() compatibility")
    except Exception as e:
        print(f"Warning: Failed to patch ControlNet: {str(e)}")

# Apply patch when module is imported
patch_controlnet_for_stream()
