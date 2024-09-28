
char_to_num = StringLookup(vocabulary=b, mask_token=None)


num_to_chars = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)