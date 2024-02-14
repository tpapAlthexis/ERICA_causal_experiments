import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import matplotlib
import math
import numpy as np
matplotlib.use('Agg')

def create_histogram(column, ax, title):
    ax.tick_params(axis='x', labelrotation=90, labelsize='small')
    ax.set_title(column.name, fontsize='small')
    stats = (title + ":\n"
             f'Mean: {column.mean():.2f}\n'
             f'StdDev: {column.std():.2f}\n'
             f'Median: {column.median():.2f}\n'
             f'Min: {column.min():.2f}\n'
             f'Max: {column.max():.2f}\n'
             f'Zeros: {np.sum(column == 0)} ({np.sum(column == 0) / len(column) * 100:.2f}%)')
    return stats

def create_csv_histograms(df):
    figures = {}
    data = {}
    for column in df.columns:
        fig = plt.figure()
        data[column] = df[column].dropna()
        ax = fig.add_subplot(111)
        data[column].hist(bins=50, ax=ax)
        title = fig.suptitle(column)
        figures[title.get_text()] = fig
    return figures, data

def gen_save_histograms(original_df, processed_df, file_name, put_in_single_page=False):
    origFig, origData = create_csv_histograms(original_df)
    procFig, procData = create_csv_histograms(processed_df)
    
    pdf_path = file_name
    with matplotlib.backends.backend_pdf.PdfPages(pdf_path) as pdf:
        if not put_in_single_page:
            for title in origFig.keys():
                # Check if there is a corresponding figure in procFig
                if title in procFig:
                    # Create a new figure
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    
                    # Plot the data from the original figure in the first subplot
                    axs[0].hist(origData[title], bins=50)
                    axs[0].set_title("Original: " + title)
                    stats = create_histogram(origData[title], axs[0], "Original")
                    axs[0].annotate(stats, xy=(0.6, 0.6), xycoords='axes fraction', fontsize='small', weight='bold')
                    
                    # Plot the data from the processed figure in the second subplot
                    axs[1].hist(procData[title], bins=50)
                    axs[1].set_title("Processed: " + title)
                    stats = create_histogram(procData[title], axs[1], "Processed")
                    axs[1].annotate(stats, xy=(0.6, 0.6), xycoords='axes fraction', fontsize='small', weight='bold')
                    
                    # Save the new figure to the PDF
                    pdf.savefig(fig)
                    plt.close(fig)
                else:
                    print(f"No matching figure for {title} in processed figures.")
        else:
            num_rows = math.ceil(len(origFig) / 2)  # Calculate the number of rows needed for the pairs of plots
            fig = plt.figure(figsize=(20, 5 * num_rows))  # Adjust the figure size based on the number of rows
            for i, title in enumerate(origFig.keys()):
                # Create subplots for the original data
                axs = fig.add_subplot(num_rows, 4, 2*i+1)  # Adjust the subplot position based on the loop index
                axs.hist(origData[title], bins=50)
                axs.set_title("Original")  # Set the title to 'Original'
                stats = create_histogram(origData[title], axs, "Original")
                axs.annotate(stats, xy=(0.6, 0.6), xycoords='axes fraction', fontsize='small', weight='bold')

                # Create subplots for the processed data
                axs = fig.add_subplot(num_rows, 4, 2*i+2)  # Adjust the subplot position based on the loop index
                axs.hist(procData[title], bins=50)
                axs.set_title("Processed")  # Set the title to 'Processed'
                stats = create_histogram(procData[title], axs, "Processed")
                axs.annotate(stats, xy=(0.6, 0.6), xycoords='axes fraction', fontsize='small', weight='bold')

            pdf.savefig(fig)  # Use pdf here
            plt.close(fig)