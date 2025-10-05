Set objFSO = CreateObject("Scripting.FileSystemObject")
Set objShell = CreateObject("WScript.Shell")

strBasePath = objFSO.GetParentFolderName(WScript.ScriptFullName)

strPythonW = objFSO.BuildPath(strBasePath, ".venv\Scripts\python.exe")
strAIScript = objFSO.BuildPath(strBasePath, ".server\server.py")

strCommand = """" & strPythonW & """ """ & strAIScript & """"

objShell.Run strCommand, 0, False