import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const root = new URL("../", import.meta.url);
let extension;
globalThis.k6TestApp = { registerExtension(value) { extension = value; } };
globalThis.k6TestApi = {};
const source = (await readFile(new URL("web/downloads.js", root), "utf8"))
    .replace('import { app } from "../../scripts/app.js";', "const app = globalThis.k6TestApp;")
    .replace('import { api } from "../../scripts/api.js";', "const api = globalThis.k6TestApi;");
const { registerDownloadUI, readEvents } = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
registerDownloadUI("kandinsky6", "Kandinsky 6");

for (const mode of ["Base", "PiFlow"]) {
    for (const input of ["Text", "Image"]) {
        const workflow = new URL(`example_workflows/K6 INT8 ${mode} ${input} to Video+Audio.json`, root);
        const graph = JSON.parse(await readFile(workflow, "utf8"));
        const note = graph.nodes.find((node) => node.type === "MarkdownNote");
        assert.equal(note?.properties.kandinsky6_download_package, "kandinsky6", fileURLToPath(workflow));
        note.widgets = [];
        note.addWidget = function (type, name, value, callback, options) {
            assert.equal(type, "button");
            assert.equal(options.serialize, false);
            this.widgets.push({ name, callback });
        };
        extension.loadedGraphNode(note);
        extension.loadedGraphNode(note);
        assert.equal(note.widgets.length, 1);
        assert.equal(note.widgets[0].name, "Download models");
    }
}

function response(chunks) {
    return new Response(new ReadableStream({ start(controller) {
        for (const chunk of chunks) controller.enqueue(new TextEncoder().encode(chunk));
        controller.close();
    } }));
}
const events = [];
const complete = response(['{"status":"down', 'loaded"}\n{"status":"complete"}\n']);
await readEvents(complete, (event) => events.push(event.status));
assert.deepEqual(events, ["downloaded", "complete"]);
assert.equal(complete.body.locked, false);
for (const chunks of [['{"status":"downloading"}\n'], ['{"status":"error","message":"disk full"}\n'], ['invalid\n']]) {
    const failed = response(chunks);
    await assert.rejects(readEvents(failed, () => {}));
    assert.equal(failed.body.locked, false);
}
console.log("PASS: four workflow buttons, idempotent binding and NDJSON stream failures");
