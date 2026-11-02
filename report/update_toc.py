"""Open the DOCX in LibreOffice, refresh the table of contents and export the PDF (DOCX left untouched)."""
import os, subprocess, sys, time
import uno
from com.sun.star.beans import PropertyValue

def prop(n, v):
    p = PropertyValue(); p.Name = n; p.Value = v; return p

src = os.path.abspath(sys.argv[1]); pdf = src[:-5] + ".pdf"
proc = subprocess.Popen(["soffice", "--headless", "--invisible", "--norestore",
                         "--accept=socket,host=localhost,port=2002;urp;"])
ctx = None
for _ in range(60):
    try:
        local = uno.getComponentContext()
        res = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
        ctx = res.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"); break
    except Exception:
        time.sleep(1)
desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(src), "_blank", 0, (prop("Hidden", True),))
idx = doc.getDocumentIndexes()
for i in range(idx.getCount()):
    idx.getByIndex(i).update()
print("updated", idx.getCount(), "index(es)")
doc.refresh()
doc.storeToURL(uno.systemPathToFileUrl(pdf), (prop("FilterName", "writer_pdf_Export"),))
doc.close(True)
try:
    desktop.terminate()
except Exception:
    pass
n = idx.getCount() if False else "TOC"
proc.wait(timeout=30)
print("exported", pdf)
