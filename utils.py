def to_cuda(data_dict):
  if data_dict is None:
    return None
  for key in data_dict.keys():
    data_dict[key] = data_dict[key].cuda()
  return data_dict
