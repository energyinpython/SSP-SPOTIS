import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyrepo_mcda.additions import rank_preferences
from ssp_spotis import SSP_SPOTIS

from pyrepo_mcda import weighting_methods as mcda_weights


def plot_lineplot(df, x_labels, y_labels, year, title, el):
    plt.figure(figsize = (9, 6))
    for k in range(df.shape[0]):
        plt.plot(x_labels, df.iloc[k, :], '.-')

        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        
        plt.annotate(y_labels[k], (x_max, df.iloc[k, -1]),
                        fontsize = 12, style='italic',
                        horizontalalignment='left')

    plt.xlabel(r'$s$' + ' coefficient value', fontsize = 12)
    plt.ylabel("Rank", fontsize = 12)
    
    plt.xticks(x_labels, fontsize = 12)
    plt.yticks(ticks=np.arange(1, len(y_labels) + 1, 1), fontsize = 12)
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle = 'dashdot')
    plt.title(title + ' compensation reduction', fontsize = 12)
    plt.tight_layout()
    plt.savefig('results/sust_coeff_' + year + '_' + el + '.png')
    plt.savefig('results/sust_coeff_' + year + '_' + el + '.eps')
    plt.savefig('results/sust_coeff_' + year + '_' + el + '.pdf')
    plt.show()



def main():

    # Load Symbols of Countries
    coun_names = pd.read_csv('./data/country_names.csv')

    # Choose evaluated year: 2020, 2021 or 2022
    year = '2021'
    df_data = pd.read_csv('./data/data_' + year + '.csv', index_col='Country')
    country_names = list(df_data.index)
    # Criteria types
    types = np.array([1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1])
    matrix = df_data.to_numpy()
    # Determine criteria weights
    weights = mcda_weights.critic_weighting(matrix)
    
    # Initialize the SSP-SPOTIS method
    ssp_spotis = SSP_SPOTIS()

    hierarchical_model = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
        [12]
    ]

    # vec_sust = np.arange(0.0, 1.1, 0.1)
    # vector with sustainability coefficient values for simulation
    vec_sust = np.arange(0.0, 0.55, 0.05)

    df_rank = pd.DataFrame(index = country_names)
    df_pref = pd.DataFrame(index = country_names)


    for vs in vec_sust:
        s = np.ones(matrix.shape[1]) * vs
        pref = ssp_spotis(matrix, weights, types, s_coeff = s)
        rank = rank_preferences(pref, reverse = False)

        df_rank["{:.2f}".format(vs)] = rank
        df_pref["{:.2f}".format(vs)] = pref

    df_pref = df_pref.rename_axis('Country')
    df_rank = df_rank.rename_axis('Country')

    df_pref.to_csv('./results/results_pref_' + year + '.csv')
    df_rank.to_csv('./results/results_rank_' + year + '.csv')


    # ==================================================================================
    # plot figure with sensitivity analysis
    plot_lineplot(df = df_rank, x_labels = vec_sust, y_labels = country_names, year = year, title = 'All criteria', el = '')

    # =============================================================================
    # hierarchical model with criteria groups
    # plot figure with sensitivity analysis for particular criteria groups

    for el, ind_name in enumerate(hierarchical_model):

        df_rank = pd.DataFrame(index = country_names)
        df_pref = pd.DataFrame(index = country_names)

        for vs in vec_sust:
            s = np.zeros(matrix.shape[1])
            s[ind_name] = vs
            pref = ssp_spotis(matrix, weights, types, s_coeff = s)
            rank = rank_preferences(pref, reverse = False)

            df_rank["{:.2f}".format(vs)] = rank
            df_pref["{:.2f}".format(vs)] = pref

        df_pref = df_pref.rename_axis('Country')
        df_rank = df_rank.rename_axis('Country')

        df_pref.to_csv('./results/results_pref_' + year + '_' + str(el + 1) + '.csv')
        df_rank.to_csv('./results/results_rank_' + year + '_' + str(el + 1) + '.csv')

        # ==================================================================================
        # plot figure with sensitivity analysis
        plot_lineplot(df = df_rank, x_labels = vec_sust, y_labels = country_names, year = year, title  = 'Criteria ' + r'$G_{' + str(el + 1) + '}$ group', el = str(el + 1))

        
if __name__ == '__main__':
    main()