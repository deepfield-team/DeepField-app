import vtk
from vtkmodules.vtkCommonCore import vtkIdTypeArray
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid, vtkSelectionNode, vtkSelection
from vtkmodules.vtkFiltersExtraction import vtkExtractSelection
from vtkmodules.vtkRenderingCore import vtkDataSetMapper
from .config import state, FIELD

colors = vtkNamedColors()

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, renderer, renderWindow):

        self.AddObserver('LeftButtonPressEvent', self.OnLeftButtonDown)
        self.AddObserver('LeftButtonReleaseEvent', self.OnLeftButtonRelease)

        self.renderer = renderer
        self.renderWindow = renderWindow
        self.picker = vtk.vtkCellPicker()
        self.selection = vtkSelection()
        self.selected_actor = None
        self.selected = vtkUnstructuredGrid()
        self.selection_node = vtkSelectionNode()
        self.selection_node.SetFieldType(vtkSelectionNode.CELL)
        self.selection_node.SetContentType(vtkSelectionNode.INDICES)

        self.extract_selection = vtkExtractSelection()
        self.selected_mapper = vtkDataSetMapper()
        self.textMapper = vtk.vtkTextMapper()
        tprop = self.textMapper.GetTextProperty()
        tprop.SetFontFamilyToArial()
        tprop.SetFontSize(18)
        tprop.ShadowOn()

        if state.theme == 'dark':
            tprop.SetColor(1, 1, 1)
        else:
            tprop.SetColor(0, 0, 0)

        tprop.SetVerticalJustificationToBottom()

        self.textActor = vtk.vtkActor2D()
        self.textActor.SetDisplayPosition(90, 50)
        self.textActor.VisibilityOn()
        self.textActor.SetMapper(self.textMapper)

    def ChangeTheme(self, theme):
        tprop = self.textMapper.GetTextProperty()
        if theme == 'light':
            tprop.SetColor(0, 0, 0)
        else:
            tprop.SetColor(1, 1, 1)

    def _AnnotatePick(self, cellId):
        if cellId != -1:
            poro = FIELD['c_data']['ROCK_PORO'][cellId]
            permx = FIELD['c_data']['ROCK_PERMX'][cellId]
            permy = FIELD['c_data']['ROCK_PERMY'][cellId]
            permz = FIELD['c_data']['ROCK_PERMZ'][cellId]
            ntg = FIELD['c_data']['ROCK_NTG'][cellId]
            self.textMapper.SetInput(f'''Cell Id: {cellId}\n\nPORO: {poro:.2f}
                                     \nPERMX: {permx:.2f}\n\nPERMY: {permy:.2f}
                                     \nPERMZ: {permz:.2f}\n\nNTG: {ntg:.2f}''')
            self.textActor.VisibilityOn()
            self.renderer.AddActor(self.textActor)

    def _HighlightCell(self, cellId):
        if cellId != -1:
            self.ids = vtkIdTypeArray()
            self.ids.SetNumberOfComponents(1)
            self.ids.InsertNextValue(cellId)
            self.selection_node.SetSelectionList(self.ids)
            self.selection.AddNode(self.selection_node)
            dataset = FIELD["main_actor"].GetMapper().GetInput()
            self.extract_selection.SetInputData(0, dataset)
            self.extract_selection.SetInputData(1, self.selection)
            self.extract_selection.Update()
            self.selected.ShallowCopy(self.extract_selection.GetOutput())
            self.selected_mapper.SetInputData(self.selected)
            self.selected_actor = vtk.vtkActor()
            self.selected_actor.SetMapper(self.selected_mapper)
            self.selected_actor.SetScale(*FIELD['scales'])
            self.selected_actor.GetProperty().SetRepresentationToWireframe()
            self.selected_actor.GetMapper().ScalarVisibilityOff()
            self.selected_actor.GetProperty().SetColor(colors.GetColor3d('Red'))
            self.selected_actor.GetProperty().SetLineWidth(4)
            self.renderer.AddActor(self.selected_actor)

    def OnLeftButtonRelease(self, obj, eventType):
        vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonUp(self)

    def OnLeftButtonDown(self, obj, eventType):
        clickPos = self.GetInteractor().GetEventPosition()
        self.picker.Pick(clickPos[0], clickPos[1], 0, self.renderer)
        cellId = self.picker.GetCellId()
        self._HighlightCell(cellId)
        self._AnnotatePick(cellId)
        vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonDown(self)