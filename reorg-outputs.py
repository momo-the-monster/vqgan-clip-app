import os
from pathlib import Path
from shutil import copyfile

def reorg(folder):
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith("output.PNG"):
                # turn into FilePath
                p = Path(filepath)
                # directory becomes guid
                guid = p.parent.name
                # parent dir becomes prefix
                prefix = p.parent.parent.name
                renamed = f"{p.parent.parent.parent.parent.absolute()}\\~collected\\{prefix}_{guid}.png"
                print (f"copying {filepath} to {renamed}")
                # DO IT
                copyfile(filepath, renamed)
                
def reorg_single(folder):
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith("output.PNG"):
                # turn into FilePath
                p = Path(filepath)
                # directory becomes guid
                guid = p.parent.name
                # parent dir becomes prefix
                prefix = p.parent.parent.name
                renamed = f"{p.parent.parent.parent.absolute()}\\~{prefix}\\{guid}.png"
                print (f"copying {filepath} to {renamed}")
                # DO IT
                copyfile(filepath, renamed)
                
# reorg("G:\\My Drive\\ai-art\\outputs")
reorg_single("G:\\My Drive\\ai-art\\outputs\\fam")