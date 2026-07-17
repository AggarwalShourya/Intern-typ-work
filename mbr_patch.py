from mbr import MBRDecoder, MBRLossFunction
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

# Save original
EncDecHybridRNNTCTCBPEModel._orig_change_decoding_strategy = EncDecHybridRNNTCTCBPEModel.change_decoding_strategy

def robust_change_decoding_strategy(self, decoding_cfg=None, decoder_type: str = None, verbose: bool = True):
    from omegaconf import OmegaConf
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecodingConfig
    from nemo.collections.asr.metrics.wer import WER

    strategy = getattr(decoding_cfg, 'strategy', None) if decoding_cfg is not None else None
    effective = decoder_type or strategy or ""

    if effective == "mbr":
        if decoding_cfg is None:
            decoding_cfg = self.cfg.decoding

        decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        # First let the original malsd strategy run so self.decoding is set up normally
        decoding_cfg_malsd = OmegaConf.merge(decoding_cfg, OmegaConf.create({"strategy": "malsd_batch"}))
        self._orig_change_decoding_strategy(decoding_cfg_malsd, decoder_type=None, verbose=False)

        # Now self.decoding is a TDTBPEDecoding with a ModifiedALSDBatchedTDTComputer inside it.
        # Find the computer and wrap its __call__ with MBR.
        mbr_loss_fn    = OmegaConf.select(decoding_cfg, "mbr_loss_fn",    default="edit_distance")
        mbr_score_norm = OmegaConf.select(decoding_cfg, "mbr_score_norm", default=True)

        # Locate ModifiedALSDBatchedTDTComputer inside self.decoding
        from nemo.collections.asr.parts.submodules.tdt_malsd_batched_computer import ModifiedALSDBatchedTDTComputer
        malsd_computer = None
        for attr in vars(self.decoding).values():
            if isinstance(attr, ModifiedALSDBatchedTDTComputer):
                malsd_computer = attr
                break

        if malsd_computer is None:
            raise RuntimeError("Could not find ModifiedALSDBatchedTDTComputer inside self.decoding. "
                               "Check what attribute holds it after strategy='malsd_tdt' is set.")

        mbr = MBRDecoder(beam_decoder=malsd_computer, loss_fn=mbr_loss_fn, score_norm=mbr_score_norm)

        # Patch __call__ on the instance so MBR runs after every MALSD decode
        import types
        def mbr_call(computer_self, x, out_len):
            batched_hyps = computer_self._orig_call(x, out_len)
            return mbr._run_mbr(batched_hyps)
        malsd_computer._orig_call = malsd_computer.__call__
        malsd_computer.__call__   = types.MethodType(mbr_call, malsd_computer)

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )
        if verbose:
            print(f"Switched decoding strategy to MBR over MALSD (loss_fn={mbr_loss_fn}, score_norm={mbr_score_norm})")
        return

    return self._orig_change_decoding_strategy(decoding_cfg, decoder_type, verbose)

EncDecHybridRNNTCTCBPEModel.change_decoding_strategy = robust_change_decoding_strategy