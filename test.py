def convert_direction(clock_dir):    
    return None if clock_dir == 0 else round((clock_dir/12)*8) % 8

# Test the function
for clock_dir in range(0, 13):
    print(f"Clock direction {clock_dir} -> Robot direction {convert_direction(clock_dir)}")