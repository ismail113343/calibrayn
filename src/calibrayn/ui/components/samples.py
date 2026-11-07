SAMPLES = (
    (
        "photosynthesis.md",
        "# Photosynthesis\n\n"
        "Photosynthesis is the process plants use to turn light energy into chemical energy. "
        "It takes place in the chloroplasts, which contain a green pigment called chlorophyll. "
        "Chlorophyll absorbs light, mostly red and blue wavelengths. The plant uses this energy "
        "to combine carbon dioxide and water into glucose and oxygen. Oxygen is released through "
        "tiny pores in the leaves called stomata.\n",
    ),
    (
        "cellular_respiration.md",
        "# Cellular respiration\n\n"
        "Cellular respiration is the process cells use to release energy from glucose. Most of it "
        "takes place in the mitochondria. Glucose reacts with oxygen to form carbon dioxide and "
        "water, and the energy released is stored in a molecule called ATP. Cells use ATP to power "
        "movement, growth and repair. Respiration and photosynthesis are linked, because the "
        "products of one are the starting materials of the other.\n",
    ),
    (
        "cell_division.md",
        "# Cell division\n\n"
        "Mitosis is the type of cell division that makes two identical cells from one. It is how "
        "organisms grow and how damaged tissue is repaired. Before mitosis begins, the cell copies "
        "its DNA so that each new cell receives a full set. Mitosis has four main stages: "
        "prophase, metaphase, anaphase and telophase. After telophase the cytoplasm splits in a "
        "step called cytokinesis.\n",
    ),
    (
        "enzymes.md",
        "# Enzymes\n\n"
        "Enzymes are proteins that speed up chemical reactions without being used up. Each enzyme "
        "has an active site with a shape that fits one kind of substrate. High temperatures or the "
        "wrong pH can change the shape of the active site, and the enzyme stops working. This is "
        "called denaturation.\n",
    ),
    (
        "protein_synthesis.md",
        "# DNA and protein synthesis\n\n"
        "DNA stores genetic information in sequences of four bases: adenine, thymine, cytosine and "
        "guanine. A gene is a section of DNA that codes for one protein. In the first step, "
        "transcription, the gene is copied into messenger RNA. In the second step, translation, "
        "ribosomes read the messenger RNA and join amino acids in the right order to build the "
        "protein.\n",
    ),
)


def sample_files():
    return [(name, text.encode("utf-8")) for name, text in SAMPLES]
