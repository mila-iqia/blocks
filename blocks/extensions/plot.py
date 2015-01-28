try:
    from bokeh.plotting import figure, output_server, show, cursession, push
    bokeh_available = True
except ImportError:
    bokeh_available = False

from block.extensions import SimpleExtension


class Plot(SimpleExtension):
    tableau20 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, document, monitors=None, open_browser=True, **kwargs):
        if not bokeh_available:
            raise ImportError
        self.plots = {}
        output_server(document)

        self.p = figure(title=document)
        if open_browser:
            show(self.p)

        kwargs.setdefault('after_every_epoch', True)
        kwargs.setdefault("before_first_epoch", True)
        super(Plot, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        iteration = log.status.iterations_done
        i = 0
        for key, value in log.current_row:
            if key not in self.plots:
                self.p.line([iteration], [value], legend=key,
                            x_axis_label='iterations', y_axis_label='value',
                            name=key, line_color=self.tableau20[i])
                i += 1
                renderer = self.p.select(dict(name=key))
                self.plots[key] = renderer[0].data_source
            else:
                self.plots[key].data['x'].append(iteration)
                self.plots[key].data['y'].append(value)
                cursession().store_objects(self.plots[key])
        push()
