from WAD_Parser.WADEditor import WADReader
import os
import envirment_utils
import ntpath

wad_directory = envirment_utils.wads + '\\'

reader = WADReader()

for wad in os.listdir(wad_directory):
    if wad.endswith(".wad") or wad.endswith(".WAD"):
        wad_with_features = reader.extract(wad_directory + wad, save_to='map-data' + '\\', root_path="map-data")
        # wad = wad_with_features["wad"]
        # levels = wad.levels
        # features = levels[3]["features"]
        # maps = levels[3]["maps"]
        #
        # print(maps)

    else:
        continue


