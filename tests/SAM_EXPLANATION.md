# From the sam3 directory
cd E:\3dhouse\geoseg\3rdparty\sam3

# Segment with default settings
python tests/sam3_annotator.py tests/images/18_132173_96779.png "building"

python tests/sam3_annotator.py tests/images/18_132173_96779.png "road"

python tests/sam3_annotator.py tests/images/18_132173_96779.png "car on the road"

python tests/sam3_annotator.py tests/images/18_132173_96779.png
python tests/sam3_annotator.py tests/images/18_132173_96780.png
python tests/sam3_annotator.py tests/images/18_132332_96774.png
python tests/sam3_annotator.py tests/images/18_132334_96778.png "road"

python tests/sam3_annotator.py tests/images

python tests/sam3_annotator.py tests/


# Simplest - just input and checkpoint
python tests/sam3_annotator.py tests/satellite/262747_173793.jpg --checkpoint ./datasets/logs/3d/checkpoints/checkpoint.pt

python tests/sam3_annotator.py tests/satellite/262747_173793.jpg --checkpoint ./models/models--facebook--sam3/blobs/9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e


# With classes
python sam3_annotator.py tests/trees "tree, grass" --checkpoint ./models/custom.pt

# With output directory
python sam3_annotator.py tests/trees "tree, grass" output_dir --checkpoint ./models/custom.pt



# sam3_mask_generator
python sam3_mask_generator.py <image_folder> <text_prompt>

# Examples
python sam3_mask_generator.py ./images "a building"
python sam3_mask_generator.py /path/to/folder "a human"
python sam3_mask_generator.py ./images "a chair"

# With score threshold (only keep masks with score >= 0.5)
python sam3_mask_generator.py ./images "a building" --score-threshold 0.5

# With custom checkpoint
python sam3_mask_generator.py ./images "a building" --checkpoint ./models/sam3.pt

python .\tests\sam3_mask_generator.py .\tests\images\20_525636_347626_768\ "a building"