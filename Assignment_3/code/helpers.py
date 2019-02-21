def get_width(current_width, filter_width, paddig_width, stride):
    return ((current_width - filter_width + 2*paddig_width)/stride) + 1

def get_height(current_height, filter_height, paddig_height, stride):
    return ((current_height - filter_height + 2*paddig_height)/stride) + 1

print(get_width(16, 3, 1, 1))
print(get_height(16, 3, 1, 1))
