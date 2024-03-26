# imports
import sys
import os
import scipy.io as sio
from PyQt6.QtWidgets import (
     QMainWindow, QApplication, QPushButton, QFileDialog, QMessageBox, QTableWidget,
     QTableWidgetItem, QVBoxLayout, QWidget, 

    )
from PyQt6.QtGui import QIntValidator
from PyQt6.QtCore import pyqtSlot, QFile, QTextStream
import numpy as np
from user_functions import (
    get_mean,
    get_mgf,
    get_variance,
    get_3rd_moment,
    get_ensemble_mean,
    get_acf_matrix,
    get_PSD,
    get_time_mean,
    get_total_power,
    autocorrelation_function,
    generate_normal_process,
    generate_normalRV,
    generate_uniform_process,
    generate_uniformRV
)
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import random
matplotlib.use("QtAgg")
from mpl_toolkits.mplot3d import Axes3D


from ui import Ui_MainWindow
import pyqtgraph as pg

class MainWindow(QMainWindow):
    
    # constructor
    def __init__(self):

        # call parent constructor
        super(MainWindow, self).__init__()

        # attach ui
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # set initial state
        self.ui.stackedWidget.setCurrentWidget(self.ui.RV_page)

        # Data storage variables
        self.RV_data = None
        self.UV_data = None
        self.NV_data = None
        self.RP_data = None
        self.Zt_data = None
        self.Wt_data = None

        # left menu buttons
        self.ui.randomVarBtn.toggled[bool].connect(self.RV_button_toggled)
        self.ui.UniformVarBtn.toggled[bool].connect(self.UV_button_toggled)
        self.ui.NormalVarBtn.toggled[bool].connect(self.NV_button_toggled)
        self.ui.RandomPrcsBtn.toggled[bool].connect(self.RP_button_toggled)
        self.ui.Z_Prcs.toggled[bool].connect(self.Zt_button_toggled)
        self.ui.W_Prcs.toggled[bool].connect(self.Wt_button_toggled)
        # TODO: Create a handler for info button and connect it to a signal

        # RV page
        self.ui.RV_btn_import.clicked.connect(self.RV_import_file)
        self.ui.RV_btn_clear.clicked.connect(self.RV_clear)

        # RP page
        self.ui.RP_btn_import.clicked.connect(self.RP_import_file)
        self.ui.RP_line_N.textEdited.connect(self.RP_display_time_mean)
        self.ui.RP_line_M.textEdited.connect(self.RP_display_M_samples)
        self.ui.RP_btn_clear.clicked.connect(self.RP_clear_button)

        # information button
        self.ui.InfoBtn.clicked.connect(self.show_info)

        # generation buttons
        self.ui.UV_generate.clicked.connect(self.save_UV)
        self.ui.NV_btn_generate.clicked.connect(self.save_NV)
        self.ui.Z_t_generate.clicked.connect(self.save_UP)
        self.ui.W_t_generate.clicked.connect(self.save_NP)

    #####################################
    ####### Left Menu Functions #########
    #####################################    

    def RV_button_toggled(self, checked):

        if checked:
            self.ui.stackedWidget.setCurrentWidget(self.ui.RV_page)

    def UV_button_toggled(self, checked):

        if checked:
            self.ui.stackedWidget.setCurrentWidget(self.ui.UV_page)

    def NV_button_toggled(self, checked):

        if checked:
            self.ui.stackedWidget.setCurrentWidget(self.ui.NV_page)

    def RP_button_toggled(self, checked):

        if checked:
            self.ui.stackedWidget.setCurrentWidget(self.ui.RP_page)

    def Zt_button_toggled(self, checked):

        if checked:
            self.ui.stackedWidget.setCurrentWidget(self.ui.Zt_page)

    def Wt_button_toggled(self, checked):

        if checked:
            self.ui.stackedWidget.setCurrentWidget(self.ui.Wt_page)

    
    
    #####################################
    ######## RV Page Functions ##########
    #####################################
            
    def RV_import_file(self):

        # prompt the user to choose the file
        path = self.get_mat_file()

        # return if not file chosen
        if not path:
            return
        
        # read the file
        mat_file = sio.loadmat(path)

        # Check if data is present in the file
        if 's' not in mat_file.keys():

            # Give user an error message
            self.invalid_input_error()

        else:
            
            # set the RV data variable
            self.RV_data = {"sample_space":mat_file['s']}

            # fill table with data
            self.RV_to_table(self.RV_data["sample_space"], self.ui.RV_table)

            # fill graphs
            mgf00, mgf01, mgf02 = self.RV_fill_graphs()

            # fill statistics
            self.RV_fill_stats(mgf00, mgf01, mgf02)
            
    def RV_clear(self):
        pass

        # clear RV data variable
        self.RV_data = None

        # clear table
        self.ui.RV_table.clear()

        # clear statistics
        self.ui.RV_out_mean.clear() # mean
        self.ui.RV_out_variance.clear() # variance
        self.ui.RV_out_3rdM.clear() # third moment
        self.ui.RV_out_M00.clear() # MGF(0)
        self.ui.RV_out_M10.clear() # MGF'(0)
        self.ui.RV_out_M20.clear() # MGF''(0)

        # clear graphs
        self.ui.RV_graph_M0.plotItem.clear()
        self.ui.RV_graph_M1.plotItem.clear()
        self.ui.RV_graph_M2.plotItem.clear()

    def RV_fill_graphs(self):

        sample_space = self.RV_data['sample_space']


        # plot MGF(t)
        mgf, t = get_mgf(sample_space)
        self.ui.RV_graph_M0.plot(t, mgf)
        self.ui.RV_graph_M0.plotItem.setTitle("MGF(t) vs. t")
        self.ui.RV_graph_M0.plotItem.setLabel('left', "MGF(t)")
        self.ui.RV_graph_M0.plotItem.setLabel('bottom', "t")
        
        # plot MGF'(t)
        mgf_1, t = get_mgf(sample_space, order=1)
        self.ui.RV_graph_M1.plot(t, mgf_1)
        self.ui.RV_graph_M1.plotItem.setTitle("MGF'(t) vs. t")
        self.ui.RV_graph_M1.plotItem.setLabel('left', "MGF'(t)")
        self.ui.RV_graph_M1.plotItem.setLabel('bottom', "t")
        

        # plot MGF''(t)
        mgf_2, t = get_mgf(sample_space, order=2)
        self.ui.RV_graph_M2.plot(t, mgf_2)
        self.ui.RV_graph_M2.plotItem.setTitle("MGF''(t) vs. t")
        self.ui.RV_graph_M2.plotItem.setLabel('left', "MGF''(t)")
        self.ui.RV_graph_M2.plotItem.setLabel('bottom', "t")

        # return all their values at t = 0
        return mgf[0], mgf_1[0], mgf_2[0]
        
    def RV_fill_stats(self, mgf00, mgf01, mgf02):
        

        sample_space = self.RV_data['sample_space']

        # mean
        self.ui.RV_out_mean.setText(str(get_mean(sample_space).round(4)))

        # variance
        self.ui.RV_out_variance.setText(str(get_variance(sample_space).round(4)))

        # third moment
        self.ui.RV_out_3rdM.setText(str(get_3rd_moment(sample_space).round(4)))

        # MGF(0)
        self.ui.RV_out_M00.setText(str(mgf00.round(4)))

        # MGF'(0)
        self.ui.RV_out_M10.setText(str(mgf01.round(4)))

        # MGF''(0)
        self.ui.RV_out_M20.setText(str(mgf02.round(4)))


    #####################################
    ######## RP Page Functions ##########
    #####################################
        
    def RP_import_file(self):

        # prompt the user to choose the file
        path = self.get_mat_file()

        # return if not file chosen
        if not path:
            return
        
        # read the file
        mat_file = sio.loadmat(path)

        # Check if data is present in the file
        if 'X' not in mat_file.keys() or 't' not in mat_file.keys():

            # Give user an error message
            self.invalid_input_error()

        else:
            
            # set the RP data variable
            self.RP_data = {"ensemble":mat_file["X"], "time":mat_file["t"]}
            
            # fill table with data
            self.RP_to_table(self.RP_data["ensemble"], self.ui.RP_table)

            # fill graphs
            self.RP_fill_graphs()

            # file stats
            self.RP_fill_stats()

    def RP_fill_graphs(self):

        # extract ensemble and time
        ensemble = self.RP_data["ensemble"]
        time = self.RP_data["time"]

        # calculate ensample mean
        ensemble_mean = get_ensemble_mean(ensemble)

        # Calculate the 2D matrix of ACF
        acf_matrix = get_acf_matrix(ensemble)

        # calculate the PSD
        f, psd = get_PSD(ensemble, time)

        # calculate time ACF
        try:
            n = int(self.ui.RP_line_N.text())
        except:
            n = 0
        lags, time_acf = autocorrelation_function(ensemble[n])
        

        # plot ensemble mean
        fig = matplotlib.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time[0], ensemble_mean[0])
        ax.set_title("Ensemble Mean vs. delay")
        ax.set_xlabel("delay")
        ax.set_ylabel("Ensemble Mean")

                # ensure a layout exists
        if not self.ui.RP_graph_ens_mean.layout():
            layout = QVBoxLayout()
            self.ui.RP_graph_ens_mean.setLayout(layout)

                # clear existing layout
        layout = self.ui.RP_graph_ens_mean.layout()
        self.clear_layout(layout)

                # write plots
        canvas = matplotlib.backends.backend_qtagg.FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)


        # plot time ACF
        fig = matplotlib.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(lags, time_acf)
        ax.set_title("ACF vs. time")
        ax.set_xlabel("time")
        ax.set_ylabel("ACF")

                # ensure a layout exists
        if not self.ui.RP_graph_time_ACF.layout():
            layout = QVBoxLayout()
            self.ui.RP_graph_time_ACF.setLayout(layout)

                # clear existing layout
        layout = self.ui.RP_graph_time_ACF.layout()
        self.clear_layout(layout)

                # write plots
        canvas = matplotlib.backends.backend_qtagg.FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)




        # plot 2D matrix of ACF
        x, y = np.meshgrid(np.arange(acf_matrix.shape[0]), np.arange(acf_matrix.shape[1]))
        fig = matplotlib.figure.Figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, acf_matrix, cmap='viridis')
        ax.set_title("3D Plot of the 2D ACF")

                # ensure a layout exists
        if not self.ui.RP_graph_stat_ACF.layout():
            layout = QVBoxLayout()
            self.ui.RP_graph_stat_ACF.setLayout(layout)

                # clear existing layout
        layout = self.ui.RP_graph_stat_ACF.layout()
        self.clear_layout(layout)

                # write plots
        canvas = matplotlib.backends.backend_qtagg.FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)



        # plot psd
        fig = matplotlib.figure.Figure()
        ax = fig.add_subplot(111)
        ax.plot(f, np.log(psd[0]))
        ax.set_title("PSD vs. frequency")
        ax.set_xlabel("frequency")
        ax.set_ylabel("PSD")

                # ensure a layout exists
        if not self.ui.RP_graph_PSD.layout():
            layout = QVBoxLayout()
            self.ui.RP_graph_PSD.setLayout(layout)

                # clear existing layout
        layout = self.ui.RP_graph_PSD.layout()
        self.clear_layout(layout)

                # write plots
        canvas = matplotlib.backends.backend_qtagg.FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)



    def RP_fill_stats(self):

        ensemble = self.RP_data['ensemble']
        time = self.RP_data['time']

        # time mean of the n-th function
        self.RP_display_time_mean()

        # Total average power
        self.ui.RP_out_total_power.setText(str(get_total_power(ensemble, time).round(2)))

        
    def RP_display_time_mean(self):

        # get n
        try:
            n = int(self.ui.RP_line_N.text())
        except:
            n = 0

        sample_function = self.RP_data['ensemble'][n, :]
        time = self.RP_data["time"]

        nth_time_mean = get_time_mean(sample_function, time[0, -1] - time[0, 0])


        self.ui.RP_out_time_mean.setText(str(nth_time_mean.round(4)))


        # time ACF at 0 delay
        lags, time_acf = autocorrelation_function(self.RP_data['ensemble'][n, :])
        self.ui.RP_out_acf.setText(str(time_acf[0].round(2)))

    def RP_display_M_samples(self):

        
        try:
            m = int(self.ui.RP_line_M.text())
        except:
            m = 0

        if m == 0:
            return
        
        ensemble = self.RP_data['ensemble']
        time = self.RP_data['time']
        
        fig = matplotlib.figure.Figure()
        
        # random signals indices
        random_indices = np.random.choice(ensemble.shape[0], size=m, replace=False)


        fig.suptitle("Multipl Sample Functions vs. time")
        for i in range(len(random_indices)):

            # plot random sample
            ax = fig.add_subplot(m, 1, i+1)
            ax.plot(time[0], ensemble[random_indices[i], :])
            ax.set_xlabel('time')
            ax.set_ylabel(f'{i}')


        # ensure a layout exists
        if not self.ui.RP_graph_samples.layout():
            layout = QVBoxLayout()
            self.ui.RP_graph_samples.setLayout(layout)

        # clear layout
        layout = self.ui.RP_graph_samples.layout()
        self.clear_layout(layout)

        # write plots
        canvas = matplotlib.backends.backend_qtagg.FigureCanvasQTAgg(fig)
        self.ui.RP_graph_samples.layout().addWidget(canvas)

    def RP_clear_button(self):

        # clear ensemble mean graph
        layout = self.ui.RP_graph_ens_mean.layout()
        self.clear_layout(layout)

        # clear PSD graph
        layout = self.ui.RP_graph_PSD.layout()
        self.clear_layout(layout)

        # clear samples graph
        layout = self.ui.RP_graph_samples.layout()
        self.clear_layout(layout)

        # clear statistical ACF graph
        layout = self.ui.RP_graph_stat_ACF.layout()
        self.clear_layout(layout)

        # clear time ACF graph
        layout = self.ui.RP_graph_time_ACF.layout()
        self.clear_layout(layout)

        # clear table
        self.ui.RP_table.clear()

        # clear linetext
        self.ui.RP_line_M.clear()
        self.ui.RP_line_N.clear()
        self.ui.RP_out_acf.clear()
        self.ui.RP_out_time_mean.clear()
        self.ui.RP_out_total_power.clear()


    #####################################
    ############ Generation #############
    #####################################
        
    def save_UV(self):
        
        try:
            a = float(self.ui.UV_a.text())
            b = float(self.ui.UV_b.text())
            
            generate_uniformRV(a, b)
            msg = QMessageBox()
            msg.setWindowTitle("")
            msg.setText("Generated successfully")
            msg.exec()
        except:
            pass


    def save_NV(self):
        try:
            a = float(self.ui.NV_input_mean.text())
            b = float(self.ui.NV_input_variance.text())
            
            generate_normalRV(a, b)

            generate_uniformRV(a, b)
            msg = QMessageBox()
            msg.setWindowTitle("")
            msg.setText("Generated successfully")
            msg.exec()
        except:
            pass

    def save_UP(self):

        try:
            a = float(self.ui.Z_t_theta_a.text())
            b = float(self.ui.Z_t_thetab.text())
            
            generate_uniform_process(a, b)

            generate_uniformRV(a, b)
            msg = QMessageBox()
            msg.setWindowTitle("")
            msg.setText("Generated successfully")
            msg.exec()
        except:
            pass



    def save_NP(self):

        try:
            mean = float(self.ui.W_t_A_mean.text())
            var = float(self.ui.W_t_A_var.text())
            
            generate_normal_process(mean, var)

            generate_uniformRV(a, b)
            msg = QMessageBox()
            msg.setWindowTitle("")
            msg.setText("Generated successfully")
            msg.exec()
        except:
            pass        

    #####################################
    ######## Utility Functions ##########
    #####################################
    
    def get_mat_file(self):
        file_filter = "Data File (*.mat)"
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select a .mat file",
            directory=os.getcwd(),
            filter=file_filter,
            initialFilter=file_filter
        )

        return response[0]
    
    def invalid_input_error(self):

        button = QMessageBox.critical(
            self,
            "Error",
            "Invalid Input!",
            buttons=QMessageBox.StandardButton.Close,
            defaultButton=QMessageBox.StandardButton.Close,
        )

    def RV_to_table(self, sample_space, table_widget : QTableWidget):

        # get a frequency array
        unique_elements, counts = np.unique(sample_space, return_counts=True)
        f_x = np.divide(counts, sample_space.size)

        # set row and column counts
        table_widget.setRowCount(len(unique_elements))
        table_widget.setColumnCount(3)

        # set column labels
        table_widget.setHorizontalHeaderLabels(['x', 'count', 'f(x)'])

        # set x
        for row in range(table_widget.rowCount()):
            item = QTableWidgetItem(str(unique_elements[row]))
            table_widget.setItem(row, 0, item)

        # set count
        for row in range(table_widget.rowCount()):
            item = QTableWidgetItem(str(counts[row]))
            table_widget.setItem(row, 1, item)

        # set f(x)
        for row in range(table_widget.rowCount()):
            item = QTableWidgetItem(str(f_x[row]))
            table_widget.setItem(row, 2, item)

    def RP_to_table(self, ensemble, table_widget : QTableWidget):


    
        # set row and column counts
        table_widget.setRowCount(ensemble.shape[0])
        table_widget.setColumnCount(ensemble.shape[1])

        # populate table with items
        for row in range(ensemble.shape[0]):
            for col in range(ensemble.shape[1]):
                item = QTableWidgetItem(str(ensemble[row, col]))
                table_widget.setItem(row, col, item)

    def clear_layout(self, layout):

        if not layout:
            return
        
        # clear the layout
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget_to_remove = item.widget()
            if widget_to_remove:
                layout.removeWidget(widget_to_remove)
        
    def show_info(self):
        
        msg = QMessageBox()
        msg.setWindowTitle("Inormation about app")
        msg.setText("Made by Mohamed Saleh CIE 19")
        msg.exec()

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
