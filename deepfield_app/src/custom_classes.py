import vtk

def annotatePick(picker, textMapper, textActor, renderer):
    if picker.GetCellId() < 0:
        textActor.VisibilityOff()
    else:
        selPt = picker.GetSelectionPoint()
        x = selPt[0]
        y = selPt[1]
        cellId = picker.GetCellId()
        cellIJK = picker.GetCellIJK()
        textMapper.SetInput(f"Cell Id is {cellId}, IJK: {cellIJK}")
        textActor.SetPosition(x, y)
        textActor.VisibilityOn()
        renderer.AddActor(textActor)

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, renderer, renderWindow):

        self.AddObserver('LeftButtonPressEvent', self.OnLeftButtonDown)
        self.AddObserver('LeftButtonReleaseEvent', self.OnLeftButtonRelease)
        #self.AddObserver('MouseMoveEvent', self.OnMouseMove)

        self.renderer = renderer
        self.chosenPiece = None
        self.renderWindow = renderWindow

        self.textMapper = vtk.vtkTextMapper()
        tprop = self.textMapper.GetTextProperty()
        tprop.SetFontFamilyToArial()
        tprop.SetFontSize(14)
        tprop.BoldOn()
        tprop.ShadowOn()
        tprop.SetColor(1, 0, 0)

        self.textActor = vtk.vtkActor2D()
        self.textActor.VisibilityOn()
        self.textActor.SetMapper(self.textMapper)


    def OnLeftButtonRelease(self, obj, eventType):
        self.chosenPiece = None
        vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonUp(self)

    def OnLeftButtonDown(self, obj, eventType):
        clickPos = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkCellPicker() #vtkPropPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.renderer)
        actor = picker.GetProp3D()#GetActor()
        self.chosenPiece = actor
        vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonDown(self)
        annotatePick(picker, self.textMapper, self.textActor, self.renderer)
    
    '''def OnMouseMove(self, obj, eventType):
        if self.chosenPiece is not None:
            mousePos = self.GetInteractor().GetEventPosition()
            self.chosenPiece.SetPosition(mousePos[0], mousePos[1], 0)
            self.renderWindow.Render(); print(self.chosenPiece.GetObjectName())
        else:
            vtk.vtkInteractorStyleTrackballCamera.OnMouseMove(self)'''