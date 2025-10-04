Set objFSO = CreateObject("Scripting.FileSystemObject")
Set objShell = CreateObject("WScript.Shell")

strBasePath = objFSO.GetParentFolderName(WScript.ScriptFullName)

strPythonW = objFSO.BuildPath(strBasePath, ".svenv\Scripts\pythonw.exe")
strAIScript = objFSO.BuildPath(strBasePath, ".pys\server.py")

strCommand = """" & strPythonW & """ """ & strAIScript & """"

objShell.Run strCommand, 0, False