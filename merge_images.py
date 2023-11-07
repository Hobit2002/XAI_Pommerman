from PIL import Image

for c in range(10):
    ims = [] 
    try:
        i = 0
        while True:
            ims.append(Image.open(f"captured/c{c}i{i}.png"))
            if i: ims[-1].putalpha(90)
            i += 1
    except FileNotFoundError:
        pass
    for im in ims[1:]:
        ims[0].paste(im, (0,0), im)
    ims[0].convert('RGB').save(f"captured/c{c}_merged.jpg")