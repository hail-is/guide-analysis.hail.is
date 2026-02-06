import numpy as np
from shiny import App, Inputs, Outputs, Session, render, ui
from datetime import date
from misc import compute_gene_contrib_to_cyto, get_safe_phe_name, gen_barplot_data
import pandas as pd

from plotting import pheno_contribs_to_lat, var_contrib_to_lat, plot_variance_components

latent_labels = []
PLOT_HEIGHT = 400
pheno_table = pd.read_csv('./pheno_table.tsv', sep='\t', index_col=0)


def set_latent_labels(ll):
    global latent_labels
    latent_labels = ll


def load_data():
    # might not need evberything
    gz = np.load('./guide_all_100lat_bl_ll.npz', allow_pickle=True)
    cos_guide = gz['cos_phe']

    pheno_guide = gz['label_phe']
    factor_guide = gz['factor_phe']  # measure pheno contrib per latent
    contrib_guide = gz['contribution_phe']  # measure latent contribution per pheno

    gene_contrib_guide = gz['contribution_gene']

    contrib_var_guide = gz['contribution_var']
    var_labels = gz['label_var']
    colors = list(pd.read_csv('./colors.txt', sep='\t', header=None)[0])

    return gz, var_labels, cos_guide, contrib_guide, contrib_var_guide, colors


def gen_data(var_labels, contrib_var_guide, var_gene_file='./snp_gene_pq.txt'):
    cyto_c_guide, names_guide, gene2cyto_dict_guide = compute_gene_contrib_to_cyto(
        var_labels, contrib_var_guide, var_gene_file
    )
    ccg = dict(zip([f'GUIDE Lat{x}' for x in range(100)], cyto_c_guide.T))
    cyto_contrib_guide = pd.DataFrame(data=ccg, index=names_guide)

    return cyto_contrib_guide, gene2cyto_dict_guide


gz, var_labels, cos_guide, contrib_guide, contrib_var_guide, colors = load_data()
cyto_contrib_guide, gene2cyto_dict_guide = gen_data(
    var_labels, contrib_var_guide, var_gene_file='./snp_gene_pq.txt'
)


plots = ui.page_fluid(
    ui.markdown(
        """
    This app is a browser for [GUIDE][0]. 
    
    Please select a phenotype of interest.
        
    [0]: https://www.dictionary.com/browse/guide
     """
    ),
    ui.layout_sidebar(
        ui.panel_sidebar(
            # ui.input_radio_buttons("cmap", "Colormap type",
            #     dict(viridis="Perceptual", gist_heat="Sequential", RdYlBu="Diverging")
            # ),
            ui.input_selectize(
                'pheno_select',
                'Choose a phenotype:',
                list(gz['label_phe']),
                selected="Alzheimer's disease/dementia",
            ),
            ui.input_slider(
                'n_var_components', 'Number of Latent Factors', min=1, max=10, value=5, step=1.0
            ),
            ui.input_slider(
                'n_phenotypes_show', 'Number of Phenotypes in Plot', min=1, max=5, value=3, step=1.0
            ),
            ui.input_slider(
                'topn_loci', 'Number of Loci in Plot', min=1, max=15, value=9, step=1.0
            ),
            ui.input_slider(
                'topn_genes', 'Number of Example Genes per Locus', min=1, max=5, value=2, step=1.0
            ),
            ui.input_slider(
                'topn_genes', 'Number of Example Genes per Locus', min=1, max=5, value=2, step=1.0
            ),
            ui.download_button('download_genes', 'Download Gene List - Top 1 Latent'),
            width=2.5,
        ),
        ui.panel_main(
            ui.row(
                ui.column(
                    3,
                    ui.output_plot('plot_variance_components_controller'),
                ),
                ui.column(9, ui.output_plot('pheno_contrib_to_lat_controller')),
            ),
            ui.row(
                ui.output_plot(
                    'var_contrib_to_lat_controller', height=PLOT_HEIGHT, fill='page_fillable'
                )
            ),
        ),
    ),
)


table = ui.page_fluid(
    ui.markdown(
        """
    This app is a browser for [GUIDE][0]. 
    
    Please select one or more phenotypes of interest for comparison.
        
    [0]: https://www.dictionary.com/browse/guide
     """
    ),
    ui.input_select(
        'selection_mode',
        'Selection mode',
        {'none': '(None)', 'single': 'Single', 'multiple': 'Multiple'},
        selected='multiple',
    ),
    ui.input_switch('fullwidth', 'Take full width', True),
    ui.input_switch('fixedheight', 'Fixed height', True),
    ui.input_switch('filters', 'Filters', True),
    ui.output_data_frame('grid'),
    ui.panel_fixed(
        ui.output_text_verbatim('detail'),
        right='10px',
        bottom='10px',
    ),
    class_='px-2',
)

app_ui = ui.page_fluid(
    ui.panel_title('GUI for GUIDE'),
    ui.navset_tab(
        ui.nav('Table', table),
        ui.nav('Bar Plots', plots),
        # ui.nav("Manhattan Plot", manplot),
    ),
    # ),
)


def server(input: Inputs, output: Outputs, session: Session):
    @output
    @render.plot
    def plot_variance_components_controller():
        interested_phenos = input.pheno_select()
        if isinstance(interested_phenos, str):
            interested_phenos = [interested_phenos]
        paths = [
            f'./all_phenos/phenotypes/guide/{get_safe_phe_name(p)}/squared_cosine_scores.tsv'
            for p in interested_phenos
        ]

        try:
            df = pd.read_csv(paths[0], sep='\t')
        except:
            return
        topn_var = input.n_var_components()
        print(f'new num var components {topn_var}')

        fig, ax, latent_labels = plot_variance_components(
            df, ['squared_cosine_score'], [interested_phenos], topn=topn_var, bar_width=0.0001
        )
        set_latent_labels(latent_labels)
        print(f'latent labels {latent_labels}')

        return fig

    @output
    @render.plot
    def var_contrib_to_lat_controller():
        interested_phenos = input.pheno_select()
        if isinstance(interested_phenos, str):
            interested_phenos = [interested_phenos]
        paths = [
            f'./all_phenos/phenotypes/guide/{get_safe_phe_name(p)}/squared_cosine_scores.tsv'
            for p in interested_phenos
        ]
        print(f'new topn genes {input.topn_genes()}')
        # latent_labels = ['GUIDE Lat72', 'GUIDE Lat10']
        fig, ax = var_contrib_to_lat(
            cyto_contrib_guide,
            latent_labels,
            spacing=3.5,
            path_to_rel_df=paths,
            topn=input.topn_loci(),
            topn_genes=input.topn_genes(),
            read_colors=True,
            gene2cyto_dict=gene2cyto_dict_guide,
        )
        return fig

    @output
    @render.plot
    def pheno_contrib_to_lat_controller():
        df = gen_barplot_data(gz['label_phe'], contrib_guide, True, colors)
        fig = pheno_contribs_to_lat(df, latent_labels, topn=input.n_phenotypes_show(), spacing=3)
        return fig

        # fig, ax = plt.subplots()
        # im = ax.imshow(data2d, cmap=input.cmap(), vmin=1, vmax=input.topn_loci())
        # fig.colorbar(im, ax=ax)

    @output
    @render.data_frame
    def grid():
        height = 300 if input.fixedheight() else None
        width = '100%' if input.fullwidth() else 'fit-content'
        return render.DataGrid(
            pheno_table,
            row_selection_mode=input.selection_mode(),
            height=height,
            width=width,
            filters=input.filters(),
        )

    @output
    @render.text
    def detail():
        print(pheno_table.iloc[1:10])
        if input.grid_selected_rows() is not None and len(input.grid_selected_rows()) > 0:
            # "split", "records", "index", "columns", "values", "table"
            return pheno_table.iloc[list(input.grid_selected_rows())]

    @session.download(
        filename=lambda: (
            f'guide-genes-{latent_labels[0].replace(" ", "_")}-{date.today().isoformat()}-{np.random.randint(100, 999)}.csv'
        )
    )
    async def download_genes():
        yield f'Gene Name\tContribution\n'
        for l, x in cyto_contrib_guide[latent_labels[0]].sort_values(ascending=False).items():
            yield f'{l}\t{x}\n'


app = App(app_ui, server, debug=True)
