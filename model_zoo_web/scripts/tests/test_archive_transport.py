"""Verify server-held model publication does not require a local model copy."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import upload_archive_model as model
from upload_archive_assets import transport_options


class ArchiveTransportTests(unittest.TestCase):
    def test_direct_transport_ignores_configured_proxy_and_jump(self):
        options=transport_options()
        self.assertIn('ProxyJump=none',options)
        self.assertIn('ProxyCommand=none',options)
        self.assertNotIn('-A',options)
        explicit=transport_options('explicit-jump')
        self.assertEqual(explicit[explicit.index('-J')+1],'explicit-jump')

    def publish(self, directory, *, existing=False, bad_source=False):
        data=b'synthetic transport fixture'
        digest=hashlib.sha256(data).hexdigest()
        artifact={'filename':'fixture.hbm','sha256':digest,'size_bytes':len(data),
                  'url':model.ARCHIVE_HTTPS+'/rdk_s600/convnext/fixture.hbm'}
        manifest=directory/'release.json'
        manifest.write_text(json.dumps({'model':{'source':'convnext'},'target':{'platform':'s600'},'metadata_revision':1,'artifact':artifact}))
        receipt=directory/'receipt.json'
        source={'exists':True,'sha256':('0'*64 if bad_source else digest),'size_bytes':len(data)}
        destination={'exists':True,'sha256':digest,'size_bytes':len(data),'directory_checksum':digest} if existing else {'exists':False}
        public={'sha256':digest,'size_bytes':len(data)}
        args=['upload_archive_model.py','--manifest',str(manifest),'--model','/server-only/no-local-model.hbm',
              '--source-host','dev-server','--ssh-host','robot@archive','--receipt',str(receipt)]
        with mock.patch.object(sys,'argv',args), mock.patch.object(model,'remote_json',side_effect=[source,destination,public])as remote, \
             mock.patch.object(model,'stream_model')as stream, mock.patch.object(model.subprocess,'check_output',side_effect=['/archive/private-stage',json.dumps({'status':'server-verified'})])as command:
            if bad_source:
                with self.assertRaisesRegex(ValueError,'Source-host model identity differs'):model.main()
                self.assertEqual(remote.call_count,1)
                stream.assert_not_called();command.assert_not_called()
                return
            model.main()
        record=json.loads(receipt.read_text())
        self.assertEqual(record['anonymous_verification_host'],'dev-server')
        self.assertEqual(record['source_path'],'/server-only/no-local-model.hbm')
        self.assertEqual(record['status'],'verified')
        self.assertEqual(remote.call_args.args[1],model.VERIFY_HTTP)
        self.assertEqual(remote.call_args.args[0][-1],'dev-server')
        self.assertEqual(stream.call_count,0 if existing else 1)
        self.assertEqual(command.call_count,0 if existing else 2)
        self.assertEqual(set(p.name for p in directory.iterdir()),{'release.json','receipt.json'})
        return record

    def test_server_source_is_streamed_and_verified_on_server(self):
        with tempfile.TemporaryDirectory()as directory:
            record=self.publish(Path(directory))
            self.assertEqual(record['transfer'],'ssh-stdio-stream; no local model file')

    def test_identical_public_model_reuses_bytes_without_a_local_copy(self):
        with tempfile.TemporaryDirectory()as directory:
            record=self.publish(Path(directory),existing=True)
            self.assertEqual(record['transfer'],'reused-identical; no local model file')

    def test_source_mismatch_stops_before_upload(self):
        with tempfile.TemporaryDirectory()as directory:self.publish(Path(directory),bad_source=True)

    def test_ssh_pipe_is_passed_directly_to_receiver(self):
        source=mock.Mock();source.stdout=io.BytesIO(b'synthetic pipe fixture');source.wait.return_value=0;source.poll.return_value=0
        with mock.patch.object(model.subprocess,'Popen',return_value=source),mock.patch.object(model.subprocess,'check_output',return_value='{"status":"stream-verified"}')as receiver:
            record=model.stream_model(['source-ssh'],'/server/model.hbm',['archive-ssh'],'/archive/stage',{'sha256':'a'*64,'size_bytes':22})
        self.assertIs(receiver.call_args.kwargs['stdin'],source.stdout)
        self.assertTrue(source.stdout.closed)
        self.assertEqual(record['status'],'stream-verified')

    def test_receiver_checks_bytes_and_removes_a_corrupt_stream(self):
        payload=b'synthetic streamed fixture';digest=hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory()as directory:
            stage=Path(directory)/'archive-private-stage';stage.touch()
            command=[sys.executable,'-c',model.RECEIVE_MODEL,str(stage),digest,str(len(payload))]
            received=subprocess.run(command,input=payload,capture_output=True)
            self.assertEqual(received.returncode,0,received.stderr)
            self.assertEqual(stage.read_bytes(),payload)
            corrupt=subprocess.run(command,input=b'truncated',capture_output=True)
            self.assertNotEqual(corrupt.returncode,0)
            self.assertFalse(stage.exists())


if __name__=='__main__':unittest.main()
