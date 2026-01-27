import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import to_hex
import pandas as pd
from matplotlib import cm
import random
import matplotlib.colors as mcolors
import matplotlib as mpl


def var_contrib_to_lat(df, cols, spacing=2.3, read_colors=True, path_to_rel_df=None, figname=None, topn=8, topn_genes=4, colors=None, gene2cyto_dict=None):
    if not gene2cyto_dict:
        print('You need a gene to cyto (pq nomenclature) dictionary.')
        return
        
    plt.clf()
    
    plt.rcParams['figure.constrained_layout.use'] = False
    mpl.rc('axes', labelsize=1, titlesize=1)

    fig, axes = plt.subplots(1, len(cols), figsize=(20, 10))
    #print(axes)
    
    if not colors:
        color_set = list(map(cm.get_cmap('Blues', topn), [1.0/topn * i for i in range(topn)]))
        colors = []
        
        #for i in range(5):
        #    for j in range(len(color_set)//5):
        #        colors.append(color_set[j+i * len(color_set)//5])

        random.Random(90).shuffle(color_set)
        color_set[0], color_set[1] = color_set[1], color_set[0]
        colors = color_set
        #print(colors)
        colors[4] = np.asarray([174, 198, 207])/255
        colors.append('lightgrey')

        #colors= ['#78a3d4', '#7ec4cf', '#9cadce', 'grey']
        
    elif colors:
        colors = colors
        colors[topn] = 'lightgrey'
        
    if read_colors:
        with open('colors.txt', 'r') as f:
            colors = []
            for l in f.read().split('\n'):
                try:
                    color = l.strip('()')
                    color = color.split(',')
                    color = list(map(lambda x: float(x.strip()), color))
                    colors.append(mcolors.to_rgb(color))
                except:
                    colors.append(color)
        colors[topn] = 'lightgrey'
        

    genes = list(df.index)
    
    middle_ys_all = []
    contribs = []
    for path in path_to_rel_df:
        try:
            contrib_df = pd.read_csv(path, 
                                        sep='\t',
                                        header=0,
                                        index_col=1) 
        except:
            return None, None
        lats = list(map(lambda x: int(x.split('GUIDE Lat')[1]), cols))
        contribs.append(list(contrib_df.loc[lats]['squared_cosine_score'] * 100))
    
    for i, lat in enumerate(cols):
        # map a cyto band to its (total contribution, [list of genes], [list of gene contributions])
        cyto_map = dict()
        for g, cont in zip(df.index, list(df[lat])):
            curr_cyto = gene2cyto_dict[g]
            # if (i == 1):
            #     print(len(genes), len(list(df[lat])))
            try:
                cyto_map[curr_cyto][0] += cont
                cyto_map[curr_cyto][1].append(g)
                cyto_map[curr_cyto][2].append(cont)
            except:
                cyto_map[curr_cyto] = [0, [g], [cont]]
            
        cyto_info = []
        for cyto, (total_cont, genes, ind_cont) in cyto_map.items():
            gene_info_sorted = sorted(zip(genes, ind_cont), key=lambda x: x[1], reverse=True)
            cyto_info.append([cyto, total_cont, gene_info_sorted])
        #if (i==1): print(cyto_info,cyto_map)
        cyto_info = sorted(cyto_info, key=lambda x: x[1], reverse=True)
        #print(cyto_info)
        to_plot = []
        ss = 0
        
        # add the top n contribution scores 
        for c in range(topn):
            #print(cyto_info)
            curr = cyto_info[c][1]
            ss += curr
            to_plot.append(curr)
        
        # add the bar at the top for 'others'
        to_plot.append(1-ss)
        bar_width=0.1
        plt.sca(axes[i])
        cs = 0
        
        #print(cyto_info[0][2][0])
        middle_ys = []
        for c in range(topn):
            #print(cyto_info[c][1])
            #total_cyto_cont_curr = cyto_info[c][1]
            label = f'{cyto_info[c][0]} ({(100*cyto_info[c][1]):.2f}%)\n'
            n_gene = 0
            curr_ind = 0
            #print(c)
            while n_gene < topn_genes:
            #for g in range(topn_genes):
                if cyto_info[c][2][curr_ind][0] != "NONE":
                    gene_name, g_cont = cyto_info[c][2][curr_ind][0], cyto_info[c][2][curr_ind][1]
                    if gene_name == "CRABP1" or gene_name == "ADAMTS7":
                        gene_name = 'CHRNA3'
                    if gene_name == 'EIF2B4':
                        gene_name =  'POMC'
                    if gene_name == 'MCM8':
                        gene_name =  'BMP2'
                    label += f' - {gene_name}\n'
                    n_gene+=1
                    curr_ind += 1
                else:
                    curr_ind += 1
                    gene_name, g_cont = cyto_info[c][2][curr_ind][0], cyto_info[c][2][curr_ind][1]
                    label += f' - {gene_name}\n'
                    curr_ind += 1
                    n_gene+=1
            label = label[:-1]
            
            middle_ys.append(cs + to_plot[c]/2)
            #axes[i].bar(f'{lat}\n({contribs[i]:.3f}%)', to_plot[c], bottom=cs, color=colors[c], label=label, width=bar_width)

            # USE THIS TO TURN CONTRIB STRINGS OFF
            #axes[i].bar(lat+'\n'+contrib_strings[i], to_plot[c], bottom=cs, color=colors[c], label=label, width=bar_width)
            print('ax label')
            axes[i].bar(0.0, to_plot[c], bottom=cs, color=colors[c], label=label, width=bar_width)

            cs += to_plot[c]

        middle_ys.append(cs + to_plot[topn]/2)
        middle_ys_all.append(middle_ys)

        # USE THIS FOR TURNING CONTRIB STRINGS OFF
        #axes[i].bar(lat+'\n'+contrib_strings[i], to_plot[topn], bottom=cs, color=colors[topn], label=f'other ({100*(1-cs):.2f}%)', width=bar_width)
        axes[i].bar(f'{lat}\n({ss*100:.2f}%)', to_plot[topn], bottom=cs, color=colors[topn], label=f'other ({100*(1-cs):.2f}%)', width=bar_width)

        #axes[i].bar(f'{lat}\n({lat_contribs[i]:.3f}%)', to_plot[topn], bottom=cs, color=colors[topn], label=f'other ({100*(1-cs):.2f}%)', width=bar_width)
        #print(f'{lat}\n({lat_contribs[i]:.3f}%)')
        plt.ylim([0, 1])

        # invert the comments here to reverse the order of the colors/labels
        #axes[i].legend(loc='center left', bbox_to_anchor=(1.05, .5, .5, 0.), fontsize="10", )
        handles, labels = axes[i].get_legend_handles_labels()
        leg = axes[i].legend(handles[::-1], labels[::-1], 
                             loc='upper left', bbox_to_anchor=(1.05, 1.2), 
                             fontsize="6")
        print('LABELS')
        print(labels)
        leg.get_frame().set_linewidth(0.0)

        #axes[i].legend().get_frame().set_linewidth(0.0)

        if i == 0:
            plt.ylabel(f'Variant Variance Components', fontsize=12)
        else:
            axes[i].set_yticklabels([])
            
        
        #print(get_ax_size(axes[i]))
            
    #plt.show()
    


    plt.subplots_adjust(wspace=spacing)
    #plt.show()
        
    for i in range(len(cols)):
        axes[i].tick_params(axis='both', which='major', labelsize=8)
        axes[i].tick_params(axis='both', which='minor', labelsize=8)
    return fig, axes


# def pheno_contribs_to_lat(df, cols, spacing=2.3, topn=3, read_colors=True, figname=None):
#     plt.clf()
#     plt.rcParams['figure.constrained_layout.use'] = False
#     nplots = len(cols)

#     fig, axes = plt.subplots(1, nplots, figsize=(20, 10))
#     #print(axes)
#     if read_colors:
#         with open('colors.txt', 'r') as f:
#             colors = []
#             for l in f.read().split('\n'):
#                 try:
#                     color = l.strip('()')
#                     color = color.split(',')
#                     color = list(map(lambda x: float(x.strip()), color))
#                     colors.append(mcolors.to_rgb(color))
#                 except:
#                     colors.append(color)
#         colors[topn] = 'lightgrey'


#     for i, lat in enumerate(cols):
#         data = df[lat].sort_values(ascending=False)
#         phenos = list(data.index)[:topn+1]
#         for p_i, p in enumerate(phenos):
#             print(textwrap.fill(p, 16))
#             phenos[p_i] = textwrap.fill(p, width=15)
                    
#         phenos = ["Forced expiratory volume in 1\nsecond (FEV1), predicted\npercentage" if "(FEV1)" in p else p for p in phenos]
#         phenos = ["Average weekly beer\nplus cider intake" if "beer" in p else p for p in phenos]
#         phenos = ["Age started wearing glasses\nor contact lenses" if "contact" in p else p for p in phenos]

#         ss = 0
#         to_plot = []
#         for c in range(topn):
#             #print(cyto_info)
#             curr = data[c]
#             ss += curr
#             to_plot.append(curr)
#         to_plot.append(1 - sum(to_plot))
        
#         # to_plot[0].append(data.iloc[0])
#         # to_plot[1].append(data.iloc[1])
#         # to_plot[2].append(data.iloc[2])
#         # to_plot[3].append(1.0-data.iloc[0]-data.iloc[1]-data.iloc[2])
        
#         print(to_plot)
#         bar_width=0.1
#         plt.sca(axes[i])

#         phenos[topn] = 'other'
#         for c in range(topn+1):
#             if c == 0:
#                 axes[i].bar(lat, to_plot[c], color=colors[c], label=phenos[c], width=bar_width)
#             else:
#                 axes[i].bar(lat, to_plot[c], bottom=sum(to_plot[:c]), color=colors[c], label=phenos[c], width=bar_width)
        
#         handles, labels = axes[i].get_legend_handles_labels()
#         leg = axes[i].legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(0.95, 1.0, 1., 0.), fontsize=6, )
#         if i == 0:
#             plt.ylabel('Phenotype Contribution Scores', fontsize=10)
#         elif i == 2:
#             axes[i].set_yticklabels([])
            
#             #plt.xlabel('Latent Label')
#         else:
#             axes[i].set_yticklabels([])
            
#         leg.get_frame().set_linewidth(0.0)
            
#     #plt.show()
    
#     plt.subplots_adjust(wspace=spacing)
#     #plt.show()
#     for i in range(len(cols)):
#         axes[i].tick_params(axis='both', which='major', labelsize=12)
#     return fig, axes


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
# Import to_hex just in case
from matplotlib.colors import to_hex 

# --- PASTE L COLOR PALETTE (6 colors for good contrast) ---
# These are soft, contrasting colors: Pink, Mint, Lavender, Peach, Light Aqua, Light Yellow
PASTEL_PALETTE = [
    '#FFADAD', # Light Coral/Pink
    '#90EE90', # Light Green/Mint
    '#B19CD9', # Light Purple/Lavender
    '#FFDAB9', # Light Peach
    '#ADD8E6', # Light Blue/Aqua
    '#FFFACD'  # Lemon Chiffon/Light Yellow
]

def pheno_contribs_to_lat(df, cols, topn=3, spacing=2.3, figname=None):
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    pheno_color_dict = {}
    top_per_lat = dict()
    phenos_seen = []
    
    # Track the index of the color used
    color_i = 0 
    
    # 2. Identify top phenotypes and assign colors from the pastel palette
    for i, lat in enumerate(cols):
        data = df[lat].sort_values(ascending=False)
        phenos = list(data.index)
        top_per_lat[lat] = phenos[:topn]
        
        for p in phenos[:topn]:
            if p not in phenos_seen:
                # Assign color by cycling through the fixed palette
                pheno_color_dict[p] = PASTEL_PALETTE[color_i % len(PASTEL_PALETTE)]
                color_i += 1
                phenos_seen.append(p)
    
    num_lats = len(cols)
    num_phenos = len(phenos_seen)
    
    phe_contrib_mat = np.zeros((num_lats, num_phenos))
    for i, lat in enumerate(cols):
        for j, pheno in enumerate(phenos_seen):
            if pheno in top_per_lat[lat]:
                try:
                    phe_contrib_mat[i, j] = df[lat][pheno]
                except:
                    phe_contrib_mat[i, j] = df[lat][pheno][1] 
            else:
                phe_contrib_mat[i, j] = 0
                
    width = 0.8
    ind = np.arange(num_lats)
    cumulative_vec = np.zeros((num_lats))
    
    # 3. Plot bars using the assigned pastel colors
    for j, pheno in enumerate(phenos_seen):
        ax.bar(ind, phe_contrib_mat[:, j], width, bottom=cumulative_vec, 
               label=pheno, color=pheno_color_dict[pheno])
        cumulative_vec += phe_contrib_mat[:, j]

    # Plot 'other' category
    other_contrib = 1 - np.sum(phe_contrib_mat, axis=1)
    other_plotted = False
    if np.any(other_contrib > 0):
        ax.bar(ind, other_contrib, width, bottom=cumulative_vec, 
               label='Other', color='lightgrey')
        phenos_seen.append('Other')
        other_plotted = True

    # 4. Handle Legend
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                    borderaxespad=0., fontsize=8, frameon=False)
    
    # Legend Customization: Ensure 'Other' key is lightgrey
    if other_plotted:
        other_idx = len(leg.get_texts()) - 1
        handles = leg.get_patches()
        
        if other_idx < len(handles):
            handles[other_idx].set_facecolor('lightgrey')
            handles[other_idx].set_edgecolor('lightgrey') 

    # 5. Set labels and titles
    ax.set_xticks(ind)
    ax.set_xticklabels(cols, fontsize=10, rotation=35, ha='right')
    ax.set_xlabel('')
    ax.set_ylabel('Phenotype Variance Components', fontsize=10)
    ax.set_ylim(0, 1)
    
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    
    if figname:
        fig.savefig(figname, bbox_inches='tight')

    return fig    

def plot_variance_components(df, cols, phenames, topn=3, read_colors=True, bar_width=0.8):
    if read_colors:
        with open('colors.txt', 'r') as f:
            colors = []
            for l in f.read().split('\n'):
                try:
                    color = l.strip('()')
                    color = color.split(',')
                    color = list(map(lambda x: float(x.strip()), color))
                    colors.append(mcolors.to_rgb(color))
                except:
                    colors.append(color)
        colors[topn] = 'lightgrey'
    plt.rcParams['figure.constrained_layout.use'] = False
    fig, axes = plt.subplots(1, len(cols), figsize=(100, 1))
    for lat in (cols):
        data = df[lat].sort_values(ascending=False)[:topn]
        to_plot = []
        ss = 0
        for c in range(topn):
            curr = data.iloc[c]
            ss += curr
            to_plot.append(curr)
        to_plot.append(1-ss)

        labels = [f'GUIDE Lat{p}' for p in list(df['PC_(zero_based)'][:topn])]
        lat_labels = labels.copy()
        labels.append('other')
        labels = [f'{p}\n({100*(to_plot[i]):.2f}%)' for i, p in enumerate(labels)]
        
        for c in range(topn+1):
            print(c)
            if c == 0:
                axes.bar(lat, to_plot[c], color=colors[c], label=labels[c], width=bar_width)
            else:
                axes.bar(lat, to_plot[c], bottom=sum(to_plot[:c]), color=colors[c], label=labels[c], width=bar_width)
        handles, labels = axes.get_legend_handles_labels()
        leg = axes.legend(handles[::-1], labels[::-1], loc='lower left', bbox_to_anchor=(1.01, 0.0, .5, 0), fontsize="8", )
        leg.get_frame().set_linewidth(0.0)
        

    axes.tick_params(axis='both', which='major', labelsize=12)
    axes.set_xticklabels(['Latent Breakdown'])
    plt.ylabel('Latent Variance Components', fontsize=12)

    return fig, axes, lat_labels
