package me.zhanghai.android.files.index

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.text.format.DateUtils
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.preference.Preference
import androidx.preference.PreferenceCategory
import androidx.preference.SwitchPreferenceCompat
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.narad.search.lib.index.enumeration.ExcludedFolders
import com.narad.search.lib.index.provision.ProvisionState
import com.narad.search.lib.index.service.BootReceiver
import com.narad.search.lib.index.worker.IndexStage
import kotlinx.coroutines.launch
import me.zhanghai.android.files.R
import me.zhanghai.android.files.filelist.FileListActivity
import me.zhanghai.android.files.ui.PreferenceFragmentCompat
import me.zhanghai.android.files.util.startActivitySafe
import me.zhanghai.android.files.util.viewModels

/** Index0: indexing status, model download, overnight toggle and excluded folders. */
class IndexingFragment : PreferenceFragmentCompat() {
    private val viewModel by viewModels { { IndexingViewModel.create(requireContext().applicationContext) } }

    private val pickFolder = registerForActivityResult(FileListActivity.OpenDirectoryContract()) { path ->
        path ?: return@registerForActivityResult
        ExcludedFolders.add(requireContext(), path.toString())
        onExclusionsChanged()
    }

    private lateinit var statusPreference: Preference
    private lateinit var modelPreference: Preference
    private lateinit var startPreference: Preference
    private lateinit var reindexPreference: Preference
    private lateinit var excludedCategory: PreferenceCategory

    override fun onCreatePreferencesFix(savedInstanceState: Bundle?, rootKey: String?) {
        val context = preferenceManager.context
        val screen = preferenceManager.createPreferenceScreen(context)
        preferenceScreen = screen

        val statusCategory = PreferenceCategory(context).apply {
            setTitle(R.string.indexing_status_category)
            isIconSpaceReserved = false
        }
        screen.addPreference(statusCategory)
        statusPreference = Preference(context).apply {
            isSelectable = false
            isIconSpaceReserved = false
        }
        modelPreference = Preference(context).apply {
            setTitle(R.string.indexing_model_title)
            isIconSpaceReserved = false
            setOnPreferenceClickListener { onModelClick(); true }
        }
        startPreference = Preference(context).apply {
            setTitle(R.string.indexing_start_title)
            isIconSpaceReserved = false
            setOnPreferenceClickListener { IndexEngine.startIndexing(); true }
        }
        reindexPreference = Preference(context).apply {
            setTitle(R.string.indexing_reindex_title)
            setSummary(R.string.indexing_reindex_summary)
            isIconSpaceReserved = false
            setOnPreferenceClickListener { IndexEngine.reindexAll(); true }
        }
        val overnightPreference = SwitchPreferenceCompat(context).apply {
            setTitle(R.string.indexing_overnight_title)
            setSummary(R.string.indexing_overnight_summary)
            isIconSpaceReserved = false
            isPersistent = false
            isChecked = BootReceiver.isOvernightEnabled(context)
            setOnPreferenceChangeListener { _, newValue ->
                BootReceiver.setOvernightEnabled(context, newValue as Boolean)
                true
            }
        }
        listOf(statusPreference, modelPreference, startPreference, reindexPreference, overnightPreference)
            .forEach { statusCategory.addPreference(it) }

        excludedCategory = PreferenceCategory(context).apply {
            setTitle(R.string.indexing_excluded_category)
            isIconSpaceReserved = false
        }
        screen.addPreference(excludedCategory)
        bindExcludedFolders()
    }

    override fun onViewCreated(view: android.view.View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch { viewModel.uiState.collect { bindStatus(it) } }
                launch { IndexEngine.provisioner.stateFlow.collect { bindModel(it) } }
            }
        }
    }

    private fun bindStatus(state: IndexingUiState) {
        val status = state.status
        statusPreference.title = when (val stage = status.stage) {
            null -> getString(
                if (status.running) R.string.indexing_status_getting_ready else R.string.indexing_status_idle
            )
            is IndexStage.WaitingForModel -> getString(R.string.indexing_status_waiting_model)
            is IndexStage.WaitingForPermission -> getString(R.string.indexing_status_waiting_permission)
            is IndexStage.Finding -> getString(R.string.indexing_status_finding, stage.kind)
            is IndexStage.Reading ->
                getString(R.string.indexing_status_reading, stage.kind, stage.done, stage.total)
            is IndexStage.Done -> getString(R.string.indexing_status_done)
            is IndexStage.Failed -> getString(R.string.indexing_status_failed, stage.reason)
            IndexStage.Paused -> getString(R.string.indexing_status_paused)
        }
        val documents = state.documentCount ?: 0
        val photos = state.photoCount ?: 0
        statusPreference.summary = status.lastRunAt?.let {
            getString(
                R.string.indexing_counts_last_run, documents, photos,
                DateUtils.getRelativeTimeSpanString(it)
            )
        } ?: getString(R.string.indexing_counts, documents, photos)
        startPreference.isEnabled = !status.running
        reindexPreference.isEnabled = !status.running
    }

    private fun bindModel(state: ProvisionState) {
        modelPreference.summary = when (state) {
            ProvisionState.NotStarted -> getString(R.string.indexing_model_not_downloaded)
            is ProvisionState.Downloading -> getString(
                R.string.indexing_model_downloading,
                if (state.bytesTotal > 0) (state.bytesDownloaded * 100 / state.bytesTotal).toInt() else 0
            )
            is ProvisionState.Verifying -> getString(R.string.indexing_model_verifying)
            ProvisionState.Ready -> getString(R.string.indexing_model_ready)
            is ProvisionState.Failed -> getString(R.string.indexing_model_failed, state.reason)
        }
        val ready = state == ProvisionState.Ready
        val access = Environment.isExternalStorageManager()
        startPreference.summary = getString(
            when {
                !ready -> R.string.indexing_start_needs_model
                !access -> R.string.indexing_start_needs_access
                else -> R.string.indexing_start_summary
            }
        )
        startPreference.isSelectable = ready && access
        reindexPreference.isVisible = ready
    }

    private fun onModelClick() {
        when (IndexEngine.provisioner.stateFlow.value) {
            ProvisionState.NotStarted, is ProvisionState.Failed -> showTermsDialog()
            else -> Unit
        }
    }

    // The only path to the model download: the Gemma terms are on screen first.
    private fun showTermsDialog() {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.indexing_terms_title)
            .setMessage(R.string.indexing_terms_message)
            .setPositiveButton(R.string.indexing_terms_accept) { _, _ -> IndexEngine.acceptTermsAndProvision() }
            .setNeutralButton(R.string.indexing_terms_link) { _, _ ->
                startActivitySafe(Intent(Intent.ACTION_VIEW, Uri.parse(GEMMA_TERMS_URL)))
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun bindExcludedFolders() {
        val context = preferenceManager.context
        excludedCategory.removeAll()
        excludedCategory.addPreference(Preference(context).apply {
            setSummary(R.string.indexing_excluded_defaults)
            isSelectable = false
            isIconSpaceReserved = false
        })
        for (folder in ExcludedFolders.userFolders(context)) {
            excludedCategory.addPreference(Preference(context).apply {
                title = folder
                isIconSpaceReserved = false
                setOnPreferenceClickListener { confirmRemove(folder); true }
            })
        }
        excludedCategory.addPreference(Preference(context).apply {
            setTitle(R.string.indexing_excluded_add)
            isIconSpaceReserved = false
            setOnPreferenceClickListener { pickFolder.launch(null); true }
        })
    }

    private fun confirmRemove(folder: String) {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.indexing_excluded_remove_title)
            .setMessage(folder)
            .setPositiveButton(R.string.indexing_excluded_remove) { _, _ ->
                ExcludedFolders.remove(requireContext(), folder)
                onExclusionsChanged()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    // A normal full pass prunes rows that are now excluded. Never a reindex: that wipes the index.
    private fun onExclusionsChanged() {
        bindExcludedFolders()
        if (IndexEngine.provisioner.stateFlow.value == ProvisionState.Ready) {
            IndexEngine.startIndexing()
        }
    }
}
