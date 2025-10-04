Set objFSO = CreateObject("Scripting.FileSystemObject")
Set objShell = CreateObject("WScript.Shell")

strBasePath = objFSO.GetParentFolderName(WScript.ScriptFullName)

strPythonW = objFSO.BuildPath(strBasePath, ".tvenv\Scripts\python.exe")
strAIScript = objFSO.BuildPath(strBasePath, ".pyt\ai_trainer.py")

strCommand = """" & strPythonW & """ """ & strAIScript & """"

objShell.Run strCommand, 0, False