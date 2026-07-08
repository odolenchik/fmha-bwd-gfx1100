import re
with open('src/fmha_bwd_kernels.hip', 'r') as f:
    lines = f.readlines()

# Find the line with 'extern "C" {'
start_idx = None
for i, line in enumerate(lines):
    if line.strip() == 'extern "C" {':
        start_idx = i
        break

if start_idx is None:
    print("Could not find extern \"C\" {")
    exit(1)

# Find the matching closing brace for the extern "C" block
brace_count = 0
end_idx = None
for i in range(start_idx, len(lines)):
    stripped = lines[i].strip()
    if stripped.startswith('extern "C" {') or stripped == 'extern "C" {':
        brace_count += 1
    elif stripped == '}':
        brace_count -= 1
        if brace_count == 0:
            end_idx = i
            break

if end_idx is None:
    print("Could not find matching closing brace for extern \"C\"")
    exit(1)

# The test functions we want to move inside are currently after the extern "C" block.
# We'll collect the lines from the current end_idx+1 to the end that are the test functions we know.
# But note: we have added test_dq_kernel, test_dk_kernel, and we already have test_dp_kernel, test_softmax_bwd_kernel, dv_kernel_sym.
# We'll just take everything from end_idx+1 to the end and move it to just before the closing brace.

# Extract the lines that are after the extern "C" block (the current test functions)
trial_lines = lines[end_idx+1:]

# We'll keep the lines up to end_idx (inclusive of the closing brace?) Actually we want to insert before the closing brace.
# So we will remove the trial_lines from the end and insert them before the line at end_idx.

new_lines = lines[:end_idx] + trial_lines + lines[end_idx:end_idx+1]  # The last part is the closing brace line.

# Write back
with open('src/fmha_bwd_kernels.hip', 'w') as f:
    f.writelines(new_lines)
