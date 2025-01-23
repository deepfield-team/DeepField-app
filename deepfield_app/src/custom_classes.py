import vtk

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, renderer, renderWindow):

        self.AddObserver('LeftButtonPressEvent', self.OnLeftButtonDown)
        self.AddObserver('LeftButtonReleaseEvent', self.OnLeftButtonRelease)
        self.AddObserver('MouseMoveEvent', self.OnMouseMove)

        self.renderer = renderer
        self.chosenPiece = None
        self.renderWindow = renderWindow

    def OnLeftButtonRelease(self, obj, eventType):
        self.chosenPiece = None
        vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonUp(self)

    def OnLeftButtonDown(self, obj, eventType):
        clickPos = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.renderer)
        actor = picker.GetActor()
        self.chosenPiece = actor
        vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonDown(self)
    
    def OnMouseMove(self, obj, eventType):
        if self.chosenPiece is not None:
            mousePos = self.GetInteractor().GetEventPosition()
            self.chosenPiece.SetPosition(mousePos[0], mousePos[1], 0)
            self.renderWindow.Render(); print(self.chosenPiece.GetObjectName())
        else:
            vtk.vtkInteractorStyleTrackballCamera.OnMouseMove(self)