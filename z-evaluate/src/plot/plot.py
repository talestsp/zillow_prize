from bokeh.charts import Histogram, Bar, BoxPlot, Scatter, show, save
from bokeh.plotting import figure, output_file
from bokeh.models import Range1d

def prediction_scatter_plot(result_df, show_plot, save_plot, title="", file_name=None):
    p = figure(plot_width=600, plot_height=600, title=title)

    p.xaxis.axis_label = "real"
    p.yaxis.axis_label = "predict"

    p.x_range = Range1d(-4, 4)
    p.y_range = Range1d(-4, 4)

    p.circle(result_df['real'], result_df['prediction'], size=3, alpha=0.2)

    if save_plot:
        output_file(file_name)
        save(p)

    if show_plot:
        show(p)
