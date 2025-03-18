import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm

def generate_venn_diagram(data, labels=None):
   
    fig, ax = plt.subplots(figsize=(8, 8))
    
    venn = venn3(subsets=data, set_labels=labels, ax=ax, layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=(1,1,1,1,1,1,1)))
    
    for region in venn.patches:
        region.set_edgecolor('black')  
        region.set_facecolor('white') 

    
    idx = 0
    for region in venn.subset_labels:
        region.set_text(str(data[idx]))
        region.set_fontsize(12)
        region.set_fontweight('bold')
        idx += 1

    
    
    plt.show()


data = (
     16,  # Only Circle A
    11,  # Only Circle B
    4,  # A and B overlap
    12,  # Only Circle C
    2,  # A and C overlap
    6,  # B and C overlap
    0,  # A, B, and C overlap
)

labels = ('Assumptions', 'Instability', 'Real-World Datasets')
generate_venn_diagram(data, labels)