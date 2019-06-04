from all_functions import import_data,expand_field,trap_knobs,set_voltages,post_process_trap
"""Run import_data and get_trap to convert to python data structuires from BEM-solver text files.
Then run expand_field and trap_knobs to construct the C file for LabRad.
Finally, run post_process_trap to calculate the trap frequencies."""

print(import_data())
####print(get_trap()) # don't need for HOA
print(expand_field())
print(trap_knobs())
print (post_process_trap())
