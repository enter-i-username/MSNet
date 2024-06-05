
def detect(x, decoder_outputs):
    y = decoder_outputs[0]
    dm = (x - y).detach()
    dm = dm ** 2
    dm = dm.sum(2)

    return dm
