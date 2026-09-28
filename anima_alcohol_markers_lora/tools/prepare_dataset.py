"""Clean the raw alcohol-marker dataset: drop unusable images, crop out props
(markers, pencils, hands, cutting mat, neighbouring sketches), inpaint small
foreign objects on plain paper, and save under sequential names.

Usage: python prepare_dataset.py <raw_root> <out_dir>
<raw_root> holds the unpacked archive folders (Животные, Фурри, Портреты с цветным лайном).
Crop boxes and inpaint rects are fractions of width/height: (left, top, right, bottom);
an inpaint entry can also be a polygon given as a list of (x, y) fractions.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

# raw file -> (output name, crop box or None, [inpaint rects / polygons])
ANIMALS = {
    "05460c6e6f3aff20d7ee6b2f8faa5125.jpg": ("tiger_lying", (0, 0.1, 1, 0.91), []),
    "080d00dcc6f9f25d0e717a9b76649fe6.jpg": ("fairy_rabbit", (0, 0, 0.97, 1), [(0.86, 0.84, 1, 1)]),
    "0c9aca2a781b1d19cfa2d1f6cd5c567c.jpg": ("husky_portrait", (0.1, 0, 1, 1), []),
    "0f77ba178a552ef1b353a61c96e1e276.jpg": ("terrier_portrait", (0.065, 0, 1, 1), []),
    "1741455138.makangeni_lupobianco7-1-25-post.jpg": ("white_wolf_profile", (0, 0.055, 1, 1), []),
    "1d06ec8f0620039e08891ce55d92e829.jpg": ("girl_with_wolf", (0, 0, 0.86, 1), []),
    "1eb46bab3d69c4d2bdc35e11d1904763.jpg": ("greyhound_scarf", None, []),
    "21fe5c93603f3d9b1b01b8a59fbe339e.jpg": ("chicken_flowers", None, []),
    "2532797cba49ad4879dd290758269cf3.jpg": ("rabbit_bow", (0, 0.055, 1, 1), []),
    "3862fcb9175738e5767ed3ea2489f8a2.jpg": ("dalmatian_portrait", (0, 0, 1, 0.83), []),
    "3a950ce6a01d02ddc1635cc425d3fcee.jpg": ("pigeons", None, [(0, 0, 0.23, 0.25), (0.76, 0.55, 0.98, 0.8)]),
    "56050d3f319d2c12b7994313d769ad9b.jpg": ("corgi_paw", (0, 0, 0.985, 1), []),
    "58df4947f4562fbbc455499f4b430ce8.jpg": ("corgi_santa_hat", None, []),
    "8389632c0c5e442df8bd6d74f629fd0a.jpg": ("small_birds", None, []),
    "8484cc88d7b81056dd9825daced40815.jpg": ("tabby_kitten", (0.165, 0.09, 0.88, 1), []),
    "89eea44e3b76de1ddd95cf08e2e136b4.jpg": ("deer_leaves", (0.15, 0, 0.79, 1), []),
    "9172845e3e3dd664a267b98b124644c8.jpg": ("teal_fox", None, []),
    "97e82eff7389823079b09ea947aa5cf9.jpg": ("dachshund_puppy", (0, 0.06, 1, 1), []),
    "9abf5bab299bed5d1fbc2e9ad56df834.jpg": ("chameleon_branch", (0, 0, 0.94, 1), []),
    "b4f3b91728803a5bfc93ff1d1dcf395c.jpg": ("rabbit_blossoms", None, [(0, 0, 0.14, 0.1)]),
    "b97d5a92ae324314a608fd86040c5707.jpg": ("blue_alpaca", (0.15, 0.02, 1, 1), [(0, 0, 0.32, 0.32)]),
    "bb5075f86d1f4989b7fd31eb3aac43c9.jpg": ("giraffe_portrait", (0, 0.03, 0.92, 0.93), [(0, 0, 0.24, 0.09), (0.66, 0, 1, 0.11)]),
    "d083dac36e718899375398a68b0b069c.jpg": ("frog_on_leaf", (0, 0.065, 1, 1), []),
    "d4fc822e2a25f2a6f962473b43851b71.jpg": ("tree_frog", None, [(0, 0, 0.065, 0.17), (0.89, 0.7, 1, 1)]),
    "d76be2ac053ee819ba7ef63b4e40daed.jpg": ("cat_teal_background", None, []),
    "ded1eecbad013afdc6a0356998fe59e9.jpg": ("shepherd_sketch", (0.25, 0, 0.885, 1), []),
    "e20645fb46d4987d43df911162475b91.jpg": ("cat_fishbowl", (0, 0, 0.99, 1), [(0, 0, 0.04, 0.04), (0.27, 0.27, 0.39, 0.4)]),
    "e48d9f144f187b8ce68e58c9998fc6f4.jpg": ("brown_bear", None, []),
    "e537ee128ce1f7eff6eb060514f30c9e.jpg": ("tabby_cat_sketch", None, []),
    "e589670bdee364fd96419330a067640d.jpg": ("owl_portrait", (0, 0.035, 1, 0.93), []),
    "fe8e95be89fa356db67167108850405a.jpg": ("burmese_cat", None, []),
    "maxresdefault (1).jpg": ("wolf_walking", (0, 0, 0.95, 1), []),
    "maxresdefault.jpg": ("blue_jay", (0.05, 0.03, 0.97, 1), []),
}

# Dropped on purpose:
#   15b6a26da7b0c92ec97a91c4095d126c.jpg - cut-out badge, big "Ghost" lettering, comb/marble photo background
#   51c84cbd0078ca26bdf62cb7eb988af8.jpg - 208x331, too small
#   684bc44c13df876b08b545cfa5e9666f.jpg - hand holding a marker covers the drawing, cannot be cropped out
#   6ad2b77d49c62728725d1ab597265a93.jpg - fineliner lies across the wolf's ear, Promarker over the corner


FURRY = {
    "0176514c59b8656d03ca213163eeb526.jpg": ("furry_pink_hair_headshot", (0, 0, 0.965, 0.73), []),
    "1633839993.keshakotolapa_белый_шум_скетч_кмм.jpg": ("furry_cat_gold_chain", None, []),
    "1640298684.dashthefox_613.jpg": ("furry_glamrock_freddy", None, [
        [(0.02, 0.08), (0.27, 0.08), (0.3, 0.12), (0.3, 0.18), (0.285, 0.198), (0.24, 0.23), (0.22, 0.32), (0.02, 0.32)],
    ]),
    "1648603299.yiffmasters_c6ef8c56-9ad3-4890-a1a0-2b59e6692f57_jpeg.jpg": ("furry_calico_scarf", None, []),
    "1651621213.dreamydogden_markers.jpg": ("furry_green_eyes_closeup", None, []),
    "1667425935.maoshan_art084.jpg": ("furry_paw_claws", (0, 0, 1, 0.93), []),
    "1707332775.khayen_andrasimg_20231123_050911823.jpg": ("furry_avian_pink_hair", (0, 0.095, 1, 1), []),
    "1707437174.khayen_puckyimg_20231124_022056080.jpg": ("furry_dog_chain_collar", None, []),
    "1720717765.shabudi_image_2024-07-11_190033837.png": ("furry_cat_green_hoodie", (0.045, 0.02, 0.905, 1), []),
    "1767400237.geckozen_1-1-26_lyndi_markers0001.jpg": ("furry_fox_girl_shorts", None, [(0, 0, 0.25, 0.09)]),
    "1788980454.heatzone_3715.jpg": ("furry_blue_shark_glasses", None, []),
    "1790288514.runarhrothgar_img_1657.jpg": ("furry_lynx_chest", None, []),
    "1790289594.runarhrothgar_img_1655.png": ("furry_black_wolf", None, []),
    "393fc9f46faf942fcddcf535d5152112.jpg": ("furry_cat_two_tone_hair", None, [(0, 0, 0.12, 0.06), (0, 0.78, 0.09, 0.92)]),
    "39775be55235eab73fe12a39b41b4fa6.jpg": ("furry_wolf_goggles", None, []),
    "6f346313299759fc3005c2d9352b22e7.jpg": ("furry_grey_badge", None, []),
    "cc94ed6fbd95fc39eedbfe30cf70c070.jpg": ("furry_rainbow_ears_drink", (0, 0, 0.94, 0.97), []),
    "d965c0164402736728dc5a1d42e37943.jpg": ("furry_horned_wolf_collar", None, []),
    "nk7simyr_lg.jpg": ("furry_fox_girl_wallpaper", None, []),
}

# Dropped from Фурри:
#   aac8456b06b9deb904b85e910e9f854f.jpg - fineliner and marker lie against the paw, inpainting leaves smears

LINE = {
    "02293923a0199039e77357e6b38ceffe.jpg": ("line_four_elf_girls", None, []),
    "0c2ff66d7771e9a5d0eb33d49c827222.jpg": ("line_girl_doodle_background", None, []),
    "2a0a5ed8a1477a9f0b8d266fd7f92482.jpg": ("line_girl_red_glasses", (0, 0, 1, 0.97), []),
    "5f4aa47ceabe4624d92e4db5f383a0b7.jpg": ("line_demon_girl_blue_hair", None, []),
    "8809f09b0f3d5f2fd8b71387d9be9689.jpg": ("line_sailor_venus", None, []),
    "88897f2246e798b5cfeffd57d9b5ffdc.jpg": ("line_snake_hair_girl", None, []),
    "9a3e4c49557d0061d62477c7845db4f9.jpg": ("line_girl_skull_cap", (0, 0, 0.84, 1), []),
    "ceacb306532b4b3e4aab47ef2c00cd1c.jpg": ("line_oni_girl_hearts", None, []),
    "dae11aaa70299d2149940b2777d6f8b3.jpg": ("line_third_eye_girl", (0.03, 0, 1, 1), []),
    "e323214cbef569548a3ae2ab1b2cb11e.jpg": ("line_three_girls", (0.01, 0.02, 1, 1), [
        [(0.585, 0), (1, 0), (1, 0.29), (0.9, 0.285), (0.72, 0.27), (0.66, 0.2), (0.6, 0.1)],
    ]),
    "e3e01d9669462f2b5a4794c27ccfd893.jpg": ("line_girl_face_stickers", None, []),
    "f6cb2c988ca18ce8c182463f5516d257.jpg": ("line_girl_hands_face", (0.27, 0, 1, 1), []),
    "scale_1200.jpeg": ("line_fox_boy_glasses", (0, 0, 0.985, 0.875), []),
}

# Dropped from Портреты с цветным лайном:
#   abe72477c82e6d67426bb1e867d0ec68.jpg - 287x527, half the frame is labelled swatches and a big signature

MAX_SIDE = 2048  # large scans are downscaled; training runs at 512-1024 anyway

GROUPS = [("Животные", ANIMALS), ("Фурри", FURRY), ("Портреты с цветным лайном", LINE)]


def px(box, w, h):
    l, t, r, b = box
    return round(l * w), round(t * h), round(r * w), round(b * h)


def process(src, crop, inpaint):
    im = Image.open(src)
    if im.mode == "RGBA":  # flatten semi-transparent scans onto white paper
        im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im)
    img = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if inpaint:
        mask = np.zeros((h, w), np.uint8)
        for shape in inpaint:
            if isinstance(shape, list):
                pts = np.array([(round(x * w), round(y * h)) for x, y in shape], np.int32)
                cv2.fillPoly(mask, [pts], 255)
            else:
                l, t, r, b = px(shape, w, h)
                mask[t:b, l:r] = 255
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
    if crop:
        l, t, r, b = px(crop, w, h)
        img = img[t:b, l:r]
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def main(raw_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    i = 0
    for folder, plan in GROUPS:
        for fname, (name, crop, inpaint) in sorted(plan.items(), key=lambda kv: kv[1][0]):
            i += 1
            im = process(os.path.join(raw_root, folder, fname), crop, inpaint)
            if max(im.size) > MAX_SIDE:
                im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            dst = os.path.join(out_dir, f"{i:03d}_{name}.png")
            im.save(dst, optimize=True)
            print(f"{dst}  {im.size[0]}x{im.size[1]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
